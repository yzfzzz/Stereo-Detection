#pragma once

#include "public.h"

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <opencv2/core/hal/interface.h>

#include <opencv2/opencv.hpp>
#include <vector>

class DepthModel {
  public:
    DepthModel();

    virtual ~DepthModel();

    // 通用的 Engine 加载和显存分配流程
    virtual void init(const std::string & engine_path, int img_w, int img_h, bool is_normalize);

    // 同步推理
    virtual std::pair<cv::Mat, cv::Mat> predict(const cv::Mat & image);
    // 异步推理
    virtual void                        predictAsync(const cv::Mat & image);
    virtual void                        waitAsync();
    virtual std::pair<cv::Mat, cv::Mat> getPredictResultAsync();

    // 子类必须实现的专属预处理
    virtual std::vector<float> preProcess(const cv::Mat & image);
    virtual void               preProcessAsync(const cv::Mat & image);
    std::vector<float>         h_mean_;
    std::vector<float>         h_std_;


  protected:
    nvinfer1::IRuntime *          runtime_;
    nvinfer1::ICudaEngine *       engine_;
    nvinfer1::IExecutionContext * context_;
    cudaStream_t                  stream_;
    std::string                   io_tensor_name_[2]{ "input", "output" };
    void *                        d_buffer_[2]{ nullptr, nullptr };

    uchar *  d_buffer_norm_depth_;     // 经过正则化的深度图，size = img_h*img_w
    uchar3 * d_buffer_norm_colormap_;  // 颜色图,size = img_h*img_w*3
    uchar *  d_buffer_dst_depth_;      // 经过正则化的深度图，size = raw_img_w*raw_img_h
    uchar3 * d_buffer_dst_colormap_;   // 颜色图,size = raw_img_w*raw_img_h*3
    float *  d_depth_infer_min_value_;
    float *  d_depth_infer_max_value_;
    void *   d_cub_mid_min_ = nullptr;
    void *   d_cub_mid_max_ = nullptr;
    size_t   cub_max_bytes_ = 0;
    size_t   cub_min_bytes_ = 0;
    size_t   cub_bytes_     = 0;
    float *  h_output_data_;
    float *  d_mean_;
    float *  d_std_;
    uchar *  d_before_preprocess_img_data_;

    int      input_h_, input_w_;      // 模型输出的尺寸
    int      raw_img_w_, raw_img_h_;  // 原始输入图像的尺寸
    uchar *  h_depth_output_data_;
    uchar3 * h_depth_colormap_data_;
    Logger   logger_;
};
