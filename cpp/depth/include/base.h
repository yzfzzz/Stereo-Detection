#pragma once

#include "public.h"

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <opencv2/core/hal/interface.h>

#include <opencv2/opencv.hpp>
#include <vector>

class BaseDepthModel {
  public:
    BaseDepthModel();

    virtual ~BaseDepthModel();

    // 通用的 Engine 加载和显存分配流程
    virtual void Init(const std::string & engine_path, int img_w, int img_h);

    // 暴露给外部调用的统一推理接口
    virtual std::pair<cv::Mat, cv::Mat> Predict(const cv::Mat & image);
    virtual void                        PredictAsync(const cv::Mat & image);
    virtual void                        WaitAsync();
    virtual std::pair<cv::Mat, cv::Mat> GetPredictResultAsync();


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
    std::string                   io_tensor_name[2]{ "input", "output" };
    void *                        buffer[2]{ nullptr, nullptr };

    uchar *  buffer_norm_depth_dev;     // 经过正则化的深度图，size = img_h*img_w
    uchar3 * buffer_norm_colormap_dev;  // 颜色图,size = img_h*img_w*3
    uchar *  buffer_dst_depth_dev;      // 经过正则化的深度图，size = origin_img_w*origin_img_h
    uchar3 * buffer_dst_colormap_dev;   // 颜色图,size = origin_img_w*origin_img_h*3
    float *  depth_infer_min_value;
    float *  depth_infer_max_value;
    void *   cub_mid_min   = nullptr;
    void *   cub_mid_max   = nullptr;
    size_t   cub_max_bytes = 0;
    size_t   cub_min_bytes = 0;
    size_t   cub_bytes     = 0;
    float *  output_data;

    int      input_h, input_w;            // 模型输出的尺寸
    int      origin_img_w, origin_img_h;  // 原始输入图像的尺寸
    uchar *  depth_output_data;
    uchar3 * depth_colormap_data;
    Logger   logger;
};
