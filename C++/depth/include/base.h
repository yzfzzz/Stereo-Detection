#pragma once

#include <cuda_runtime_api.h>
#include <NvInfer.h>

#include <opencv2/opencv.hpp>
#include <vector>

class BaseDepthModel {
  public:
    BaseDepthModel();

    virtual ~BaseDepthModel();

    // 通用的 Engine 加载和显存分配流程
    virtual void Init(const std::string & engine_path, nvinfer1::ILogger & logger);

    // 暴露给外部调用的统一推理接口
    virtual std::pair<cv::Mat, cv::Mat> Predict(const cv::Mat & image);

  protected:
    // 子类必须实现的专属预处理
    virtual std::vector<float> Preprocess(const cv::Mat & image) = 0;

    // 可选的后处理接口，默认实现是直接返回原始深度图，颜色图
    virtual std::pair<cv::Mat, cv::Mat> Postprocess();

  protected:
    nvinfer1::IRuntime *          runtime;
    nvinfer1::ICudaEngine *       engine;
    nvinfer1::IExecutionContext * context;
    cudaStream_t                  stream;
    void *                        buffer[2]{ nullptr, nullptr };
    int                           input_h, input_w;            // 模型输出的尺寸
    int                           origin_img_w, origin_img_h;  // 原始输入图像的尺寸
    float *                       output_data;
};
