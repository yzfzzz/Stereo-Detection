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
    virtual void init(const std::string & engine_path, int img_w, int img_h);

    // 同步推理
    virtual std::pair<cv::Mat, cv::Mat> predict(const cv::Mat & image);
    // 异步推理
    virtual void                        predictAsync(const cv::Mat & image);
    virtual void                        waitAsync();
    virtual std::pair<cv::Mat, cv::Mat> getPredictResultAsync();

    // 子类必须实现的专属预处理
    virtual std::vector<float> preProcess(const cv::Mat & image) = 0;
    virtual void               preProcessAsync(const cv::Mat & image);
    float                      mean_host_[3] = { 0, 0, 0 };
    float                      std_host_[3]  = { 1.0f, 1.0f, 1.0f };


  protected:
    nvinfer1::IRuntime *          runtime_;
    nvinfer1::ICudaEngine *       engine_;
    nvinfer1::IExecutionContext * context_;
    cudaStream_t                  stream_;
    std::string                   io_tensor_name_[2]{ "input", "output" };
    void *                        buffer_[2]{ nullptr, nullptr };

    uchar *  buffer_norm_depth_dev_;     // 经过正则化的深度图，size = img_h*img_w
    uchar3 * buffer_norm_colormap_dev_;  // 颜色图,size = img_h*img_w*3
    uchar * buffer_dst_depth_dev_;  // 经过正则化的深度图，size = origin_img_w*origin_img_h
    uchar3 * buffer_dst_colormap_dev_;  // 颜色图,size = origin_img_w*origin_img_h*3
    float *  depth_infer_min_value_;
    float *  depth_infer_max_value_;
    void *   cub_mid_min_   = nullptr;
    void *   cub_mid_max_   = nullptr;
    size_t   cub_max_bytes_ = 0;
    size_t   cub_min_bytes_ = 0;
    size_t   cub_bytes_     = 0;
    float *  output_data_;
    float *  mean_dev_;
    float *  std_dev_;
    uchar *  before_preprocess_img_data_dev_;

    int      input_h_, input_w_;            // 模型输出的尺寸
    int      origin_img_w_, origin_img_h_;  // 原始输入图像的尺寸
    uchar *  depth_output_data_;
    uchar3 * depth_colormap_data_;
    Logger   logger_;
};
