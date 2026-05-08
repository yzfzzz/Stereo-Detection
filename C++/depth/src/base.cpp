#include "base.h"
#include "scope_timer.h"
#include <fstream>
#include <iostream>
#include <memory>

// TensorRT Logger 也可以放在通用的地方
class Logger : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char * msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[Depth TensorRT] " << msg << std::endl;
        }
    }
};

BaseDepthModel::BaseDepthModel() :
    runtime(nullptr),
    engine(nullptr),
    context(nullptr),
    stream(nullptr),
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
#endif
}

void BaseDepthModel::Init(const std::string & engine_path, nvinfer1::ILogger & logger) {
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

#if defined(__aarch64__) || defined(__arm__)
    stream = nullptr;
#else
    cudaStreamCreate(&stream);
#endif
    cudaMalloc(&buffer[0], 3 * input_h * input_w * sizeof(float));
    cudaMalloc(&buffer[1], input_h * input_w * sizeof(float));
    output_data = new float[input_h * input_w];
}

std::pair<cv::Mat, cv::Mat> BaseDepthModel::Predict(const cv::Mat & image) {
    // 1. 预处理
    std::vector<float> input = Preprocess(image);

    cudaMemcpyAsync(buffer[0], input.data(), 3 * input_h * input_w * sizeof(float), cudaMemcpyHostToDevice, stream);

 {   
    ScopedTimer timer("3-2.DepthModel Infer");
    // 2. 推理
#if NV_TENSORRT_MAJOR < 10
    context->enqueueV2(buffer, stream, nullptr);

#else
    context->setTensorAddress(io_tensor_name[0].c_str(), buffer[0]);
    context->setTensorAddress(io_tensor_name[1].c_str(), buffer[1]);

    bool status = context->enqueueV3(stream);
    if (!status) {
        std::cerr << "TensorRT enqueueV3 failed!" << std::endl;
        return {};
    }

#endif
 }

    // 3. 后处理：只拷贝数据，返回原始深度矩阵
    cudaMemcpyAsync(output_data, buffer[1], input_h * input_w * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::pair<cv::Mat, cv::Mat> postprocess_result = Postprocess();
    return postprocess_result;
}

std::pair<cv::Mat, cv::Mat> BaseDepthModel::Postprocess() {
    ScopedTimer timer("3-3.BaseDepthModel::Postprocess");
    cv::Mat depth_mat(input_h, input_w, CV_32FC1, output_data);
    cv::normalize(depth_mat, depth_mat, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::Mat colormap;
    cv::applyColorMap(depth_mat, colormap, cv::COLORMAP_INFERNO);
    cv::resize(colormap, colormap, cv::Size(origin_img_w, origin_img_h));
    return std::make_pair(depth_mat, colormap);
}
