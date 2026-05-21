#include "base.h"
#include <cassert>
#include <cstdint>
#include <cub/cub.cuh>
#include "postprocess.h"
#include "public.h"
#include "scope_timer.h"

#include <opencv2/core/hal/interface.h>

#include <fstream>
#include <iostream>
#include <memory>

BaseDepthModel::BaseDepthModel() :
    runtime(nullptr),
    engine(nullptr),
    context(nullptr),
    stream(0),
    input_h(0),
    input_w(0),
    output_data(nullptr) {}

BaseDepthModel::~BaseDepthModel() {
    if (stream) {
        cudaStreamDestroy(stream);
    }
    if (buffer[0]) {
        cudaFree(buffer[0]);
    }
    if (buffer[1]) {
        cudaFree(buffer[1]);
    }
    if (output_data) {
        delete[] output_data;
    }

    if(buffer_norm_depth_dev) {
        cudaFree(buffer_norm_depth_dev);
    }
    if(buffer_norm_colormap_dev){
        cudaFree(buffer_norm_colormap_dev);
    }
    if(buffer_dst_depth_dev){
        cudaFree(buffer_dst_depth_dev);
    }
    if(buffer_dst_colormap_dev){
        cudaFree(buffer_dst_colormap_dev);
    }
    if(depth_infer_max_value){
        cudaFree(depth_infer_max_value);
    }
    if(depth_infer_min_value){
        cudaFree(depth_infer_min_value);
    }

#if NV_TENSORRT_MAJOR < 10
    if (context) {
        context->destroy();
    }
    if (engine) {
        engine->destroy();
    }
    if (runtime) {
        runtime->destroy();
    }

    if(buffer_norm_depth_dev) {
        cudaFree(buffer_norm_depth_dev);
    }
    if(buffer_norm_colormap_dev){
        cudaFree(buffer_norm_colormap_dev);
    }
    if(buffer_dst_depth_dev){
        cudaFree(buffer_dst_depth_dev);
    }
    if(buffer_dst_colormap_dev){
        cudaFree(buffer_dst_colormap_dev);
    }
    if(depth_infer_max_value){
        cudaFree(depth_infer_max_value);
    }
    if(depth_infer_min_value){
        cudaFree(depth_infer_min_value);
    }
#endif
}

void BaseDepthModel::Init(const std::string & engine_path, int img_w, int img_h) {
    origin_img_h = img_h;
    origin_img_w = img_w;
    assert(origin_img_h > 0 && origin_img_w > 0 && "Invalid image dimensions");
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

    runtime = nvinfer1::createInferRuntime(logger);
    engine  = runtime->deserializeCudaEngine(engineData.get(), modelSize);
    context = engine->createExecutionContext();

    // 兼容 TensorRT 8.x
#if NV_TENSORRT_MAJOR < 10
    int nb_bindings = engine->getNbBindings();
    assert(nb_bindings == 2 && "Expecting exactly 2 bindings (input and output)");
    io_tensor_name[0] = engine->getBindingName(0);
    io_tensor_name[1] = engine->getBindingName(1);
    // Define input dimensions
    auto input_dims   = engine->getBindingDimensions(0);
    input_h           = input_dims.d[2];
    input_w           = input_dims.d[3];

#else
    int nb_bindings = engine->getNbIOTensors();
    assert(nb_bindings == 2 && "Expecting exactly 2 bindings (input and output)");
    io_tensor_name[0] = engine->getIOTensorName(0);
    io_tensor_name[1] = engine->getIOTensorName(1);
    auto input_dims   = engine->getTensorShape(io_tensor_name[0].c_str());
    input_h           = input_dims.d[2];
    input_w           = input_dims.d[3];
#endif

    cudaStreamCreate(&stream);

    cudaMalloc(&buffer[0], 3 * input_h * input_w * sizeof(float));
    cudaMalloc(&buffer[1], input_h * input_w * sizeof(float));
    cudaMalloc((void **) &buffer_norm_depth_dev, input_h * input_w * sizeof(uchar));
    cudaMalloc((void **) &buffer_norm_colormap_dev, input_h * input_w * sizeof(uchar3));
    cudaMalloc((void **) &buffer_dst_depth_dev, origin_img_h * origin_img_w * sizeof(uchar));
    cudaMalloc((void **) &buffer_dst_colormap_dev, origin_img_h * origin_img_w * sizeof(uchar3));
    cudaMalloc((void **) &depth_infer_min_value, sizeof(float));
    cudaMalloc((void **) &depth_infer_max_value, sizeof(float));
    // 查询 CUB 所需临时显存大小

    cub::DeviceReduce::Min(nullptr, cub_min_bytes, (float*)buffer[1], depth_infer_min_value, input_h * input_w, stream);
    cub::DeviceReduce::Max(nullptr, cub_max_bytes, (float*)buffer[1], depth_infer_max_value, input_h * input_w, stream);
    cub_bytes = std::max(cub_min_bytes, cub_max_bytes);
    cudaMalloc((void **) &cub_mid_min, cub_bytes);
    cudaMalloc((void **) &cub_mid_max, cub_bytes);
    output_data = new float[input_h * input_w];

    depth_output_data   = new uchar[origin_img_h * origin_img_w];
    depth_colormap_data = new uchar3[origin_img_h * origin_img_w];

    context->setTensorAddress(io_tensor_name[0].c_str(), buffer[0]);
    context->setTensorAddress(io_tensor_name[1].c_str(), buffer[1]);

    initColorMapTable();
}

std::pair<cv::Mat, cv::Mat> BaseDepthModel::Predict(const cv::Mat & image) {

    std::vector<float> input = Preprocess(image);
    cudaMemcpy(buffer[0], input.data(), 3 * input_h * input_w * sizeof(float), cudaMemcpyHostToDevice);

    bool status = context->executeV2(buffer);
    if (!status) {
        std::cerr << "TensorRT enqueueV3 failed!" << std::endl;
        return {};
    }

    cudaMemcpy(output_data, buffer[1], input_h * input_w * sizeof(float), cudaMemcpyDeviceToHost);
    cv::Mat depth_mat(input_h, input_w, CV_32FC1, output_data);
    cv::normalize(depth_mat, depth_mat, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::Mat colormap;
    cv::applyColorMap(depth_mat, colormap, cv::COLORMAP_INFERNO);
    cv::resize(colormap, colormap, cv::Size(origin_img_w, origin_img_h));
    return std::make_pair(depth_mat, colormap);
}


void BaseDepthModel::PredictAsync(const cv::Mat & image) {
    std::vector<float> input = Preprocess(image);
    // 数据异步拷贝至 GPU
    CHECK_CUDA(cudaMemcpyAsync(buffer[0], input.data(), 3 * input_h * input_w * sizeof(float), cudaMemcpyHostToDevice, stream));

#if NV_TENSORRT_MAJOR < 10
    context->enqueueV2(buffer, stream, nullptr);

#else
    context->enqueueV3(stream);
#endif

    cub::DeviceReduce::Min(cub_mid_min, cub_bytes, (float*)buffer[1], depth_infer_min_value, input_h * input_w, stream);
    cub::DeviceReduce::Max(cub_mid_max, cub_bytes, (float*)buffer[1], depth_infer_max_value, input_h * input_w, stream);

    normalize_colormap_resize((float*)buffer[1], buffer_norm_depth_dev, buffer_norm_colormap_dev, buffer_dst_depth_dev,
                              buffer_dst_colormap_dev, depth_infer_min_value, depth_infer_max_value, input_w, input_h,
                              origin_img_w, origin_img_h, stream);

    CHECK_CUDA(cudaMemcpyAsync(depth_output_data, buffer_dst_depth_dev, origin_img_h * origin_img_w * sizeof(uchar),
                    cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemcpyAsync(depth_colormap_data, buffer_dst_colormap_dev, origin_img_h * origin_img_w * sizeof(uchar3),
                    cudaMemcpyDeviceToHost, stream));
}

void BaseDepthModel::WaitAsync() {
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

std::pair<cv::Mat, cv::Mat> BaseDepthModel::GetPredictResultAsync() { 
    auto                        depth_output       = cv::Mat(origin_img_h, origin_img_w, CV_8UC1, depth_output_data);
    auto                        depth_colormap     = cv::Mat(origin_img_h, origin_img_w, CV_8UC3, depth_colormap_data);
    std::pair<cv::Mat, cv::Mat> result = std::make_pair(depth_output, depth_colormap);
    return result;
}


