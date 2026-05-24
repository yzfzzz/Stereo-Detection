#include "base.h"
#include "postprocess.h"
#include "preprocess.h"
#include "public.h"

#include <opencv2/core/hal/interface.h>

#include <cassert>
#include <cub/cub.cuh>
#include <fstream>
#include <iostream>
#include <memory>

BaseDepthModel::BaseDepthModel() :
    runtime_(nullptr),
    engine_(nullptr),
    context_(nullptr),
    stream_(0),
    input_h_(0),
    input_w_(0),
    output_data_(nullptr) {}

BaseDepthModel::~BaseDepthModel() {
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
    if (buffer_[0]) {
        cudaFree(buffer_[0]);
    }
    if (buffer_[1]) {
        cudaFree(buffer_[1]);
    }
    if (output_data_) {
        delete[] output_data_;
    }

    if (buffer_norm_depth_dev_) {
        cudaFree(buffer_norm_depth_dev_);
    }
    if (buffer_norm_colormap_dev_) {
        cudaFree(buffer_norm_colormap_dev_);
    }
    if (buffer_dst_depth_dev_) {
        cudaFree(buffer_dst_depth_dev_);
    }
    if (buffer_dst_colormap_dev_) {
        cudaFree(buffer_dst_colormap_dev_);
    }
    if (depth_infer_max_value_) {
        cudaFree(depth_infer_max_value_);
    }
    if (depth_infer_min_value_) {
        cudaFree(depth_infer_min_value_);
    }
    if (before_preprocess_img_data_dev_) {
        cudaFree(before_preprocess_img_data_dev_);
    }

#if NV_TENSORRT_MAJOR < 10
    if (context_) {
        context_->destroy();
    }
    if (engine_) {
        engine_->destroy();
    }
    if (runtime_) {
        runtime_->destroy();
    }

    if (buffer_norm_depth_dev_) {
        cudaFree(buffer_norm_depth_dev_);
    }
    if (buffer_norm_colormap_dev_) {
        cudaFree(buffer_norm_colormap_dev_);
    }
    if (buffer_dst_depth_dev_) {
        cudaFree(buffer_dst_depth_dev_);
    }
    if (buffer_dst_colormap_dev_) {
        cudaFree(buffer_dst_colormap_dev_);
    }
    if (depth_infer_max_value_) {
        cudaFree(depth_infer_max_value_);
    }
    if (depth_infer_min_value_) {
        cudaFree(depth_infer_min_value_);
    }
#endif
}

void BaseDepthModel::init(const std::string & engine_path, int img_w, int img_h) {
    origin_img_h_ = img_h;
    origin_img_w_ = img_w;
    assert(origin_img_h_ > 0 && origin_img_w_ > 0 && "Invalid image dimensions");
    std::ifstream engineStream(engine_path, std::ios::binary);
    if (!engineStream.is_open()) {
        std::cerr << "Failed to open engine file: " << engine_path << std::endl;
        return;
    }
    engineStream.seekg(0, std::ios::end);
    size_t modelSize = engineStream.tellg();
    engineStream.seekg(0, std::ios::beg);
    std::unique_ptr<char[]> engineData(new char[modelSize]);
    engineStream.read(engineData.get(), modelSize);
    engineStream.close();

    runtime_ = nvinfer1::createInferRuntime(logger_);
    engine_  = runtime_->deserializeCudaEngine(engineData.get(), modelSize);
    context_ = engine_->createExecutionContext();

    // 兼容 TensorRT 8.x
#if NV_TENSORRT_MAJOR < 10
    int nb_bindings = engine_->getNbBindings();
    assert(nb_bindings == 2 && "Expecting exactly 2 bindings (input and output)");
    io_tensor_name_[0] = engine_->getBindingName(0);
    io_tensor_name_[1] = engine_->getBindingName(1);
    // Define input dimensions
    auto input_dims    = engine_->getBindingDimensions(0);
    input_h_           = input_dims.d[2];
    input_w_           = input_dims.d[3];

#else
    int nb_bindings = engine_->getNbIOTensors();
    assert(nb_bindings == 2 && "Expecting exactly 2 bindings (input and output)");
    io_tensor_name_[0] = engine_->getIOTensorName(0);
    io_tensor_name_[1] = engine_->getIOTensorName(1);
    auto input_dims    = engine_->getTensorShape(io_tensor_name_[0].c_str());
    input_h_           = input_dims.d[2];
    input_w_           = input_dims.d[3];
#endif

    cudaStreamCreate(&stream_);

    cudaMalloc(&buffer_[0], 3 * input_h_ * input_w_ * sizeof(float));
    cudaMalloc(&buffer_[1], input_h_ * input_w_ * sizeof(float));
    cudaMalloc((void **) &buffer_norm_depth_dev_, input_h_ * input_w_ * sizeof(uchar));
    cudaMalloc((void **) &buffer_norm_colormap_dev_, input_h_ * input_w_ * sizeof(uchar3));
    cudaMalloc((void **) &depth_infer_min_value_, sizeof(float));
    cudaMalloc((void **) &depth_infer_max_value_, sizeof(float));
    cudaMalloc((void **) &mean_dev_, 3 * sizeof(float));
    cudaMalloc((void **) &std_dev_, 3 * sizeof(float));
    cudaMemcpy(mean_dev_, mean_host_, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(std_dev_, std_host_, 3 * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_CUDA(
        cudaMalloc((void **) &before_preprocess_img_data_dev_, 3 * origin_img_h_ * origin_img_w_ * sizeof(uchar)));

    // 查询 CUB 所需临时显存大小
    cub::DeviceReduce::Min(nullptr, cub_min_bytes_, (float *) buffer_[1], depth_infer_min_value_, input_h_ * input_w_,
                           stream_);
    cub::DeviceReduce::Max(nullptr, cub_max_bytes_, (float *) buffer_[1], depth_infer_max_value_, input_h_ * input_w_,
                           stream_);
    cub_bytes_ = std::max(cub_min_bytes_, cub_max_bytes_);
    cudaMalloc((void **) &cub_mid_min_, cub_bytes_);
    cudaMalloc((void **) &cub_mid_max_, cub_bytes_);
    output_data_ = new float[input_h_ * input_w_];

#if NV_TENSORRT_MAJOR >= 10
    // TRT 8.x: 不需要显式 setTensorAddress，enqueueV2 会按 binding 顺序读取 buffer 数组
    // TRT 10.x: 必须显式设置 Tensor 地址
    context_->setTensorAddress(io_tensor_name_[0].c_str(), buffer_[0]);
    context_->setTensorAddress(io_tensor_name_[1].c_str(), buffer_[1]);
#endif

    depth_output_data_   = new uchar[origin_img_h_ * origin_img_w_];
    depth_colormap_data_ = new uchar3[origin_img_h_ * origin_img_w_];
#if defined(__aarch64__) && defined(ENABLE_JESTON_MEM_MANAGED)
    // Jeston 上使用统一内存
    cudaMallocManaged(&buffer_dst_depth_dev_, origin_img_h_ * origin_img_w_ * sizeof(uchar));
    cudaMallocManaged(&buffer_dst_colormap_dev_, origin_img_h_ * origin_img_w_ * sizeof(uchar3));
#else
    cudaMalloc((void **) &buffer_dst_depth_dev_, origin_img_h_ * origin_img_w_ * sizeof(uchar));
    cudaMalloc((void **) &buffer_dst_colormap_dev_, origin_img_h_ * origin_img_w_ * sizeof(uchar3));
#endif

    initColorMapTable();
}

std::pair<cv::Mat, cv::Mat> BaseDepthModel::predict(const cv::Mat & image) {
    std::vector<float> input = preProcess(image);
    cudaMemcpy(buffer_[0], input.data(), 3 * input_h_ * input_w_ * sizeof(float), cudaMemcpyHostToDevice);

    bool status = context_->executeV2(buffer_);
    if (!status) {
        std::cerr << "TensorRT enqueueV3 failed!" << std::endl;
        return {};
    }

    cudaMemcpy(output_data_, buffer_[1], input_h_ * input_w_ * sizeof(float), cudaMemcpyDeviceToHost);
    cv::Mat depth_mat(input_h_, input_w_, CV_32FC1, output_data_);
    cv::normalize(depth_mat, depth_mat, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::Mat colormap;
    cv::applyColorMap(depth_mat, colormap, cv::COLORMAP_INFERNO);
    cv::resize(colormap, colormap, cv::Size(origin_img_w_, origin_img_h_));
    return std::make_pair(depth_mat, colormap);
}

void BaseDepthModel::predictAsync(const cv::Mat & image) {
    // 数据异步拷贝至 GPU, 并进行 cuda 前处理
    preProcessAsync(image);

#if defined(__aarch64__) || defined(__arm__) || NV_TENSORRT_MAJOR < 10
    // For Jetson Nano (ARM64) and older TensorRT versions
    bool status = context_->enqueueV2(buffer_, stream_, nullptr);

#else
    bool status = context_->enqueueV3(stream_);
#endif
    if (!status) {
        std::cerr << "TensorRT enqueue failed!" << std::endl;
        return;
    }

    cub::DeviceReduce::Min(cub_mid_min_, cub_bytes_, (float *) buffer_[1], depth_infer_min_value_, input_h_ * input_w_,
                           stream_);
    cub::DeviceReduce::Max(cub_mid_max_, cub_bytes_, (float *) buffer_[1], depth_infer_max_value_, input_h_ * input_w_,
                           stream_);

    normalize_colormap_resize((float *) buffer_[1], buffer_norm_depth_dev_, buffer_norm_colormap_dev_,
                              buffer_dst_depth_dev_, buffer_dst_colormap_dev_, depth_infer_min_value_,
                              depth_infer_max_value_, input_w_, input_h_, origin_img_w_, origin_img_h_, stream_);

    CHECK_CUDA(cudaMemcpyAsync(depth_output_data_, buffer_dst_depth_dev_, origin_img_h_ * origin_img_w_ * sizeof(uchar),
                               cudaMemcpyDeviceToHost, stream_));
    CHECK_CUDA(cudaMemcpyAsync(depth_colormap_data_, buffer_dst_colormap_dev_,
                               origin_img_h_ * origin_img_w_ * sizeof(uchar3), cudaMemcpyDeviceToHost, stream_));
}

void BaseDepthModel::waitAsync() {
    CHECK_CUDA(cudaStreamSynchronize(stream_));
}

std::pair<cv::Mat, cv::Mat> BaseDepthModel::getPredictResultAsync() {
    auto                        depth_output   = cv::Mat(origin_img_h_, origin_img_w_, CV_8UC1, depth_output_data_);
    auto                        depth_colormap = cv::Mat(origin_img_h_, origin_img_w_, CV_8UC3, depth_colormap_data_);
    std::pair<cv::Mat, cv::Mat> result         = std::make_pair(depth_output, depth_colormap);
    return result;
}

void BaseDepthModel::preProcessAsync(const cv::Mat & image) {
    CHECK_CUDA(cudaMemcpyAsync(before_preprocess_img_data_dev_, image.data,
                               3 * origin_img_h_ * origin_img_w_ * sizeof(uchar), cudaMemcpyHostToDevice, stream_));
    depthPreprocess(before_preprocess_img_data_dev_, (float *) buffer_[0], origin_img_w_, origin_img_h_, input_w_,
                    input_h_, mean_dev_, std_dev_, stream_);
}
