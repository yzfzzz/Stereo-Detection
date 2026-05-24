#include "infer.h"

#include "calibrator.h"
#include "postprocess.h"
#include "preprocess.h"
#include "public.h"
#include "utils.h"

#include <NvOnnxParser.h>

#include <cassert>
#include <fstream>
#include <iostream>

YoloDetector::YoloDetector(const std::string trtFile,
                           int               raw_img_w,
                           int               raw_img_h,
                           int               gpuId,
                           float             nmsThresh,
                           float             confThresh,
                           int               numClass) :
    trtFile_(trtFile),
    nmsThresh_(nmsThresh),
    confThresh_(confThresh),
    numClass_(numClass),
    raw_img_w_(raw_img_w),
    raw_img_h_(raw_img_h) {
    g_logger_ = Logger(nvinfer1::ILogger::Severity::kERROR);
    cudaSetDevice(gpuId);

    CHECK_CUDA(cudaStreamCreate(&stream_));

    // load engine
    getEngine();

    context_ = engine_->createExecutionContext();
    // Set input dimensions - use appropriate API based on platform/TensorRT version
#if defined(__aarch64__) || defined(__arm__) || NV_TENSORRT_MAJOR < 10
    // For Jetson Nano (ARM64) and older TensorRT versions
    context_->setBindingDimensions(0, nvinfer1::Dims{
                                          4, { 1, 3, input_h_, input_w_ }
    });

    // Get output dimensions using binding index
    nvinfer1::Dims outDims = engine_->getBindingDimensions(1);  // [1, 84, 8400]
#else
    // For newer TensorRT versions on x86_64
    context_->setInputShape("images", nvinfer1::Dims{
                                          4, { 1, 3, input_h_, input_w_ }
    });

    // Get output dimensions using tensor name
    nvinfer1::Dims outDims =
        context_->getTensorShape(engine_->getIOTensorName(1));  // [1, 84, 8400]
#endif

    // yolov8: [1, 30, 8400]
    // yolo26: [1, 300, 6]
    OUTPUT_CANDIDATES_ = outDims.d[2];
    int outputSize     = 1;  //yolov8 = 84*8400, yolov26 = 6*300
    for (int i = 0; i < outDims.nbDims; i++) {
        outputSize *= outDims.d[i];
    }
    if (OUTPUT_CANDIDATES_ == 6) {
        // 走yolo26推理，输出候选框较少，且已经经过nms处理，不需要再做一次nms了
        is_need_nms_                = false;
        yolo26_max_num_output_bbox_ = outDims.d[1];
        yolo26_num_box_element_     = outDims.d[2];
        std::cout << "Running YOLOv26 inference, output candidates num: " << OUTPUT_CANDIDATES_
                  << std::endl;
    }

    // prepare output data space on host
    if (!is_need_nms_) {
        output_data_ = new float[yolo26_max_num_output_bbox_ * yolo26_num_box_element_];
    } else {
        output_data_ = new float[1 + MAX_NUM_OUTPUT_BBOX * NUM_BOX_ELEMENT];
    }
    // prepare input and output space on device
    v_buffer_d_.resize(2, nullptr);
    CHECK_CUDA(cudaMalloc(&v_buffer_d_[0], 3 * input_h_ * input_w_ * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&v_buffer_d_[1], outputSize * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&transpose_device_, outputSize * sizeof(float)));
    CHECK_CUDA(
        cudaMalloc(&decode_device_, (1 + MAX_NUM_OUTPUT_BBOX * NUM_BOX_ELEMENT) * sizeof(float)));

    CHECK_CUDA(cudaMalloc((void **) &src_dev_data_, sizeof(uchar) * raw_img_h_ * raw_img_w_ * 3));
    CHECK_CUDA(cudaMalloc((void **) &mid_dev_data_, sizeof(uchar) * input_h_ * input_w_ * 3));
}

void YoloDetector::getEngine() {
    if (access(trtFile_.c_str(), F_OK) == 0) {
        std::ifstream engineFile(trtFile_, std::ios::binary);
        long int      fsize = 0;

        engineFile.seekg(0, engineFile.end);
        fsize = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);
        std::vector<char> engineString(fsize);
        engineFile.read(engineString.data(), fsize);
        if (engineString.size() == 0) {
            std::cout << "Failed getting serialized engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded getting serialized engine!" << std::endl;

        runtime_ = nvinfer1::createInferRuntime(g_logger_);
        engine_  = runtime_->deserializeCudaEngine(engineString.data(), fsize);
#if NV_TENSORRT_MAJOR < 10
        // Define input dimensions
        auto input_dims = engine_->getBindingDimensions(0);
        input_h_        = input_dims.d[2];
        input_w_        = input_dims.d[3];
#else
        auto input_dims = engine_->getTensorShape(engine_->getIOTensorName(0));
        input_h_        = input_dims.d[2];
        input_w_        = input_dims.d[3];
#endif
        assert(input_h_ > 0 && input_w_ > 0 &&
               "Input dimensions must be positive! Check engine binding or dynamic shape setting.");

        if (engine_ == nullptr) {
            std::cout << "Failed loading engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded loading engine!" << std::endl;
    } else {
        nvinfer1::IBuilder *           builder = nvinfer1::createInferBuilder(g_logger_);
        nvinfer1::INetworkDefinition * network = builder->createNetworkV2(
            1U << int(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        nvinfer1::IOptimizationProfile * profile = builder->createOptimizationProfile();
        nvinfer1::IBuilderConfig *       config  = builder->createBuilderConfig();

        nvinfer1::IInt8Calibrator * pCalibrator = nullptr;
        if (B_FP16_MODE) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        if (B_INT8_MODE) {
            config->setFlag(nvinfer1::BuilderFlag::kINT8);
            int batchSize = 8;
            pCalibrator   = new Int8EntropyCalibrator2(
                batchSize, input_w_, input_h_, CALIBRATION_DATA_PATH.c_str(), CACHE_FILE.c_str());
            config->setInt8Calibrator(pCalibrator);
        }

        nvonnxparser::IParser * parser = nvonnxparser::createParser(*network, g_logger_);
        if (!parser->parseFromFile(ONNX_FILE.c_str(), int(g_logger_.reportable_severity_))) {
            std::cout << std::string("Failed parsing .onnx file!") << std::endl;
            for (int i = 0; i < parser->getNbErrors(); ++i) {
                auto * error = parser->getError(i);
                std::cout << std::to_string(int(error->code())) << std::string(":")
                          << std::string(error->desc()) << std::endl;
            }
            return;
        }
        std::cout << std::string("Succeeded parsing .onnx file!") << std::endl;

        nvinfer1::ITensor * inputTensor = network->getInput(0);
        profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kMIN,
                               nvinfer1::Dims{
                                   4, { 1, 3, input_h_, input_w_ }
        });
        profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kOPT,
                               nvinfer1::Dims{
                                   4, { 1, 3, input_h_, input_w_ }
        });
        profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kMAX,
                               nvinfer1::Dims{
                                   4, { 1, 3, input_h_, input_w_ }
        });
        config->addOptimizationProfile(profile);

        nvinfer1::IHostMemory * engineString = builder->buildSerializedNetwork(*network, *config);
        std::cout << "Succeeded building serialized engine!" << std::endl;

        runtime_ = nvinfer1::createInferRuntime(g_logger_);
        engine_  = runtime_->deserializeCudaEngine(engineString->data(), engineString->size());
        if (engine_ == nullptr) {
            std::cout << "Failed building engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded building engine!" << std::endl;

        if (B_INT8_MODE && pCalibrator != nullptr) {
            delete pCalibrator;
        }

        std::ofstream engineFile(trtFile_, std::ios::binary);
        engineFile.write(static_cast<char *>(engineString->data()), engineString->size());
        std::cout << "Succeeded saving .plan file!" << std::endl;

        delete engineString;
        delete parser;
        delete config;
        delete network;
        delete builder;
    }
}

YoloDetector::~YoloDetector() {
    cudaStreamDestroy(stream_);

    for (int i = 0; i < 2; ++i) {
        CHECK_CUDA(cudaFree(v_buffer_d_[i]));
    }

    CHECK_CUDA(cudaFree(transpose_device_));
    CHECK_CUDA(cudaFree(decode_device_));
    CHECK_CUDA(cudaFree(src_dev_data_));
    CHECK_CUDA(cudaFree(mid_dev_data_));

    delete[] output_data_;

    delete context_;
    delete engine_;
    delete runtime_;
}

std::vector<Detection> YoloDetector::inference(const cv::Mat & img) {
    if (img.empty()) {
        return {};
    }

    // put input on device, then letterbox、bgr to rgb、hwc to chw、normalize.
    preprocess(img, (float *) v_buffer_d_[0], src_dev_data_, mid_dev_data_, raw_img_h_, raw_img_w_,
               input_h_, input_w_, stream_);
    cudaStreamSynchronize(stream_);

    // TensorRT inference - use appropriate API based on platform/TensorRT version
    void * bingding_buffers[2] = { v_buffer_d_[0], v_buffer_d_[1] };
    bool   status              = context_->executeV2(bingding_buffers);
    if (!status) {
        std::cerr << "TensorRT enqueueV3 failed!" << std::endl;
        return {};
    }

    if (!is_need_nms_) {
        // 走yolo26推理，输出候选框较少，且已经经过nms处理，不需要再做一次nms了
        // [1 1801]
        CHECK_CUDA(
            cudaMemcpy(output_data_, v_buffer_d_[1],
                       (yolo26_max_num_output_bbox_ * yolo26_num_box_element_) * sizeof(float),
                       cudaMemcpyDeviceToHost));

    } else {
        // 走yolo8推理，输出候选框较多，需要做一次nms处理

        // transpose [1 84 8400] convert to [1 8400 84]
        transpose((float *) v_buffer_d_[1], transpose_device_, OUTPUT_CANDIDATES_, numClass_ + 4,
                  stream_);
        // convert [1 8400 84] to [1 7001]
        decode(transpose_device_, decode_device_, OUTPUT_CANDIDATES_, numClass_, confThresh_,
               MAX_NUM_OUTPUT_BBOX, NUM_BOX_ELEMENT, stream_);
        // cuda nms
        nms(decode_device_, nmsThresh_, MAX_NUM_OUTPUT_BBOX, NUM_BOX_ELEMENT, stream_);
        cudaStreamSynchronize(stream_);

        CHECK_CUDA(cudaMemcpy(output_data_, decode_device_,
                              (1 + MAX_NUM_OUTPUT_BBOX * NUM_BOX_ELEMENT) * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    return postProcess(output_data_, img);
}

std::vector<Detection> YoloDetector::postProcess(float * outputData, const cv::Mat & img) {
    std::vector<Detection> vDetections;
    int                    count;
    if (!is_need_nms_) {
        count = std::min(yolo26_max_num_output_bbox_, MAX_NUM_OUTPUT_BBOX);
    } else {
        count = std::min((int) outputData[0], MAX_NUM_OUTPUT_BBOX);
    }
    for (int i = 0; i < count; i++) {
        int       pos;
        Detection det;
        auto      get_effective_detection = [&]() {
            memcpy(det.bbox.data(), &outputData[pos], 4 * sizeof(float));
            det.conf    = outputData[pos + 4];
            det.classId = (int) outputData[pos + 5];
            vDetections.push_back(det);
        };
        if (!is_need_nms_) {
            pos = i * yolo26_num_box_element_;
            if (outputData[pos + 4] > confThresh_) {
                get_effective_detection();
            }

        } else {
            pos          = 1 + i * NUM_BOX_ELEMENT;
            int keepFlag = (int) outputData[pos + 6];
            if (keepFlag == 1) {
                get_effective_detection();
            }
        }
    }

    for (size_t j = 0; j < vDetections.size(); j++) {
        scale_bbox(img, vDetections[j].bbox.data(), input_w_, input_h_);
    }

    return vDetections;
}

void YoloDetector::inferenceAsync(const cv::Mat & img) {
    if (img.empty()) {
        return;
    }

    preprocess(img, (float *) v_buffer_d_[0], src_dev_data_, mid_dev_data_, raw_img_h_, raw_img_w_,
               input_h_, input_w_, stream_);

#if defined(__aarch64__) || defined(__arm__) || NV_TENSORRT_MAJOR < 10
    // For Jetson Nano (ARM64) and older TensorRT versions
    void * bindings[] = { v_buffer_d_[0], v_buffer_d_[1] };
    bool   status     = context_->enqueueV2(bindings, stream, nullptr);
#else
    // For newer TensorRT versions on x86_64
    context_->setTensorAddress("images", v_buffer_d_[0]);
    context_->setTensorAddress("output0", v_buffer_d_[1]);
    bool status = context_->enqueueV3(stream_);
#endif

    if (!status) {
        std::cerr << "TensorRT enqueue failed!" << std::endl;
        return;
    }

    if (!is_need_nms_) {
        // 走yolo26推理，输出候选框较少，且已经经过nms处理，不需要再做一次nms了
        // [1 1801]
        CHECK_CUDA(
            cudaMemcpyAsync(output_data_, v_buffer_d_[1],
                            (yolo26_max_num_output_bbox_ * yolo26_num_box_element_) * sizeof(float),
                            cudaMemcpyDeviceToHost, stream_));
    } else {
        // 走yolo8推理，输出候选框较多，需要做一次nms处
        // transpose [1 84 8400] convert to [1 8400 84]
        transpose((float *) v_buffer_d_[1], transpose_device_, OUTPUT_CANDIDATES_, numClass_ + 4,
                  stream_);
        // convert [1 8400 84] to [1 7001]
        decode(transpose_device_, decode_device_, OUTPUT_CANDIDATES_, numClass_, confThresh_,
               MAX_NUM_OUTPUT_BBOX, NUM_BOX_ELEMENT, stream_);
        // cuda nms
        nms(decode_device_, nmsThresh_, MAX_NUM_OUTPUT_BBOX, NUM_BOX_ELEMENT, stream_);

        CHECK_CUDA(cudaMemcpyAsync(output_data_, decode_device_,
                                   (1 + MAX_NUM_OUTPUT_BBOX * NUM_BOX_ELEMENT) * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream_));
    }
}

void YoloDetector::waitAsync() {
    cudaStreamSynchronize(stream_);
}

std::vector<Detection> YoloDetector::getInferResultAsync(const cv::Mat & img) {
    return postProcess(output_data_, img);
}

void YoloDetector::drawImage(cv::Mat & img, std::vector<Detection> & inferResult) {
    // draw inference result on image
    for (size_t i = 0; i < inferResult.size(); i++) {
        cv::Scalar bboxColor(get_random_int(), get_random_int(), get_random_int());
        cv::Rect   r(round(inferResult[i].bbox[0]), round(inferResult[i].bbox[1]),
                     round(inferResult[i].bbox[2] - inferResult[i].bbox[0]),
                     round(inferResult[i].bbox[3] - inferResult[i].bbox[1]));
        cv::rectangle(img, r, bboxColor, 2);

        std::string className = V_CLASS_NAMES[(int) inferResult[i].classId];
        std::string labelStr  = className + " " + std::to_string(inferResult[i].conf).substr(0, 4);

        cv::Size  textSize = cv::getTextSize(labelStr, cv::FONT_HERSHEY_PLAIN, 1.2, 2, NULL);
        cv::Point topLeft(r.x, r.y - textSize.height - 3);
        cv::Point bottomRight(r.x + textSize.width, r.y);
        cv::rectangle(img, topLeft, bottomRight, bboxColor, -1);
        cv::putText(img, labelStr, cv::Point(r.x, r.y - 2), cv::FONT_HERSHEY_PLAIN, 1.2,
                    cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    }
}
