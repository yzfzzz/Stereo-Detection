#include "infer.h"

#include "calibrator.h"
#include "postprocess.h"
#include "preprocess.h"
#include "utils.h"

#include <NvOnnxParser.h>

#include <cassert>
#include <fstream>
#include <iostream>

using namespace nvinfer1;

YoloDetector::YoloDetector(const std::string trtFile, int gpuId, float nmsThresh, float confThresh, int numClass) :
    trtFile_(trtFile),
    nmsThresh_(nmsThresh),
    confThresh_(confThresh),
    numClass_(numClass) {
    gLogger = Logger(ILogger::Severity::kERROR);
    cudaSetDevice(gpuId);

    CHECK(cudaStreamCreate(&stream));

    // load engine
    get_engine();

    context = engine->createExecutionContext();
    // Set input dimensions - use appropriate API based on platform/TensorRT version
#if defined(__aarch64__) || defined(__arm__) || NV_TENSORRT_MAJOR < 10
    // For Jetson Nano (ARM64) and older TensorRT versions
    context->setBindingDimensions(0, nvinfer1::Dims{
                                         4, { 1, 3, input_h, input_w }
    });

    // Get output dimensions using binding index
    nvinfer1::Dims outDims = engine->getBindingDimensions(1);  // [1, 84, 8400]
#else
    // For newer TensorRT versions on x86_64
    context->setInputShape("images", nvinfer1::Dims{
                                         4, { 1, 3, input_h, input_w }
    });

    // Get output dimensions using tensor name
    nvinfer1::Dims outDims = context->getTensorShape(engine->getIOTensorName(1));  // [1, 84, 8400]
#endif

    // yolov8: [1, 30, 8400]
    // yolo26: [1, 300, 6]
    OUTPUT_CANDIDATES = outDims.d[2];
    int outputSize    = 1;  //yolov8 = 84*8400, yolov26 = 6*300
    for (int i = 0; i < outDims.nbDims; i++) {
        outputSize *= outDims.d[i];
    }
    if (OUTPUT_CANDIDATES == 6) {
        // 走yolo26推理，输出候选框较少，且已经经过nms处理，不需要再做一次nms了
        is_need_nms_               = false;
        yolo26_max_num_output_bbox = outDims.d[1];
        yolo26_num_box_element     = outDims.d[2];
        std::cout << "Running YOLOv26 inference, output candidates num: " << OUTPUT_CANDIDATES << std::endl;
    }

    // prepare output data space on host
    if (!is_need_nms_) {
        outputData = new float[yolo26_max_num_output_bbox * yolo26_num_box_element];
    } else {
        outputData = new float[1 + kMaxNumOutputBbox * kNumBoxElement];
    }
    // prepare input and output space on device
    vBufferD.resize(2, nullptr);
    CHECK(cudaMalloc(&vBufferD[0], 3 * input_h * input_w * sizeof(float)));
    CHECK(cudaMalloc(&vBufferD[1], outputSize * sizeof(float)));

    CHECK(cudaMalloc(&transposeDevice, outputSize * sizeof(float)));
    CHECK(cudaMalloc(&decodeDevice, (1 + kMaxNumOutputBbox * kNumBoxElement) * sizeof(float)));
}

void YoloDetector::get_engine() {
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

        runtime = createInferRuntime(gLogger);
        engine  = runtime->deserializeCudaEngine(engineString.data(), fsize);
#if NV_TENSORRT_MAJOR < 10
        // Define input dimensions
        auto input_dims = engine->getBindingDimensions(0);
        input_h         = input_dims.d[2];
        input_w         = input_dims.d[3];
#else
        auto input_dims = engine->getTensorShape(engine->getIOTensorName(0));
        input_h         = input_dims.d[2];
        input_w         = input_dims.d[3];
#endif
        assert(input_h > 0 && input_w > 0 &&
               "Input dimensions must be positive! Check engine binding or dynamic shape setting.");

        if (engine == nullptr) {
            std::cout << "Failed loading engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded loading engine!" << std::endl;
    } else {
        IBuilder *           builder = createInferBuilder(gLogger);
        INetworkDefinition * network =
            builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        IOptimizationProfile * profile = builder->createOptimizationProfile();
        IBuilderConfig *       config  = builder->createBuilderConfig();
        // Set workspace size - use appropriate API based on platform/TensorRT version
#if defined(__aarch64__) || defined(__arm__) || NV_TENSORRT_MAJOR < 10
        // For Jetson Nano (ARM64) and older TensorRT versions
        config->setMaxWorkspaceSize(1 << 30);  // 1 GB
#else
        // For newer TensorRT versions on x86_64
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 30);  // 1 GB
#endif
        IInt8Calibrator * pCalibrator = nullptr;
        if (bFP16Mode) {
            config->setFlag(BuilderFlag::kFP16);
        }
        if (bINT8Mode) {
            config->setFlag(BuilderFlag::kINT8);
            int batchSize = 8;
            pCalibrator =
                new Int8EntropyCalibrator2(batchSize, input_w, input_h, calibrationDataPath.c_str(), cacheFile.c_str());
            config->setInt8Calibrator(pCalibrator);
        }

        nvonnxparser::IParser * parser = nvonnxparser::createParser(*network, gLogger);
        if (!parser->parseFromFile(onnxFile.c_str(), int(gLogger.reportableSeverity))) {
            std::cout << std::string("Failed parsing .onnx file!") << std::endl;
            for (int i = 0; i < parser->getNbErrors(); ++i) {
                auto * error = parser->getError(i);
                std::cout << std::to_string(int(error->code())) << std::string(":") << std::string(error->desc())
                          << std::endl;
            }
            return;
        }
        std::cout << std::string("Succeeded parsing .onnx file!") << std::endl;

        ITensor * inputTensor = network->getInput(0);
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN,
                               nvinfer1::Dims{
                                   4, { 1, 3, input_h, input_w }
        });
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT,
                               nvinfer1::Dims{
                                   4, { 1, 3, input_h, input_w }
        });
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX,
                               nvinfer1::Dims{
                                   4, { 1, 3, input_h, input_w }
        });
        config->addOptimizationProfile(profile);

        IHostMemory * engineString = builder->buildSerializedNetwork(*network, *config);
        std::cout << "Succeeded building serialized engine!" << std::endl;

        runtime = createInferRuntime(gLogger);
        engine  = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
        if (engine == nullptr) {
            std::cout << "Failed building engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded building engine!" << std::endl;

        if (bINT8Mode && pCalibrator != nullptr) {
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
    cudaStreamDestroy(stream);

    for (int i = 0; i < 2; ++i) {
        CHECK(cudaFree(vBufferD[i]));
    }

    CHECK(cudaFree(transposeDevice));
    CHECK(cudaFree(decodeDevice));

    delete[] outputData;

    delete context;
    delete engine;
    delete runtime;
}

std::vector<Detection> YoloDetector::inference(cv::Mat & img) {
    if (img.empty()) {
        return {};
    }

    // put input on device, then letterbox、bgr to rgb、hwc to chw、normalize.
    preprocess(img, (float *) vBufferD[0], input_h, input_w, stream);

    // TensorRT inference - use appropriate API based on platform/TensorRT version
#if defined(__aarch64__) || defined(__arm__) || NV_TENSORRT_MAJOR < 10
    // For Jetson Nano (ARM64) and older TensorRT versions
    void * bindings[] = { vBufferD[0], vBufferD[1] };
    bool   status     = context->enqueueV2(bindings, stream, nullptr);
#else
    // For newer TensorRT versions on x86_64
    context->setTensorAddress("images", vBufferD[0]);
    context->setTensorAddress("output0", vBufferD[1]);
    bool status = context->enqueueV3(stream);
#endif
    if (!status) {
        std::cerr << "TensorRT enqueueV3 failed!" << std::endl;
        return {};
    }
    if (!is_need_nms_) {
        // 走yolo26推理，输出候选框较少，且已经经过nms处理，不需要再做一次nms了
        // [1 1801]
        CHECK(cudaMemcpyAsync(outputData, vBufferD[1],
                              (yolo26_max_num_output_bbox * yolo26_num_box_element) * sizeof(float),
                              cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
    } else {
        // 走yolo8推理，输出候选框较多，需要做一次nms处理
        // transpose [1 84 8400] convert to [1 8400 84]
        transpose((float *) vBufferD[1], transposeDevice, OUTPUT_CANDIDATES, numClass_ + 4, stream);
        // convert [1 8400 84] to [1 7001]
        decode(transposeDevice, decodeDevice, OUTPUT_CANDIDATES, numClass_, confThresh_, kMaxNumOutputBbox,
               kNumBoxElement, stream);
        // cuda nms
        nms(decodeDevice, nmsThresh_, kMaxNumOutputBbox, kNumBoxElement, stream);

        CHECK(cudaMemcpyAsync(outputData, decodeDevice, (1 + kMaxNumOutputBbox * kNumBoxElement) * sizeof(float),
                              cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
    }

    std::vector<Detection> vDetections;
    int                    count;
    if (!is_need_nms_) {
        count = std::min(yolo26_max_num_output_bbox, kMaxNumOutputBbox);
    } else {
        count = std::min((int) outputData[0], kMaxNumOutputBbox);
    }
    for (int i = 0; i < count; i++) {
        int       pos;
        Detection det;
        auto      get_effective_detection = [&]() {
            memcpy(det.bbox, &outputData[pos], 4 * sizeof(float));
            det.conf    = outputData[pos + 4];
            det.classId = (int) outputData[pos + 5];
            vDetections.push_back(det);
        };
        if (!is_need_nms_) {
            pos = i * yolo26_num_box_element;
            if (outputData[pos + 4] > confThresh_) {
                get_effective_detection();
            }

        } else {
            pos          = 1 + i * kNumBoxElement;
            int keepFlag = (int) outputData[pos + 6];
            if (keepFlag == 1) {
                get_effective_detection();
            }
        }
    }

    for (size_t j = 0; j < vDetections.size(); j++) {
        scale_bbox(img, vDetections[j].bbox, input_w, input_h);
    }

    return vDetections;
}

void YoloDetector::draw_image(cv::Mat & img, std::vector<Detection> & inferResult) {
    // draw inference result on image
    for (size_t i = 0; i < inferResult.size(); i++) {
        cv::Scalar bboxColor(get_random_int(), get_random_int(), get_random_int());
        cv::Rect   r(round(inferResult[i].bbox[0]), round(inferResult[i].bbox[1]),
                     round(inferResult[i].bbox[2] - inferResult[i].bbox[0]),
                     round(inferResult[i].bbox[3] - inferResult[i].bbox[1]));
        cv::rectangle(img, r, bboxColor, 2);

        std::string className = vClassNames[(int) inferResult[i].classId];
        std::string labelStr  = className + " " + std::to_string(inferResult[i].conf).substr(0, 4);

        cv::Size  textSize = cv::getTextSize(labelStr, cv::FONT_HERSHEY_PLAIN, 1.2, 2, NULL);
        cv::Point topLeft(r.x, r.y - textSize.height - 3);
        cv::Point bottomRight(r.x + textSize.width, r.y);
        cv::rectangle(img, topLeft, bottomRight, bboxColor, -1);
        cv::putText(img, labelStr, cv::Point(r.x, r.y - 2), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(255, 255, 255), 2,
                    cv::LINE_AA);
    }
}
