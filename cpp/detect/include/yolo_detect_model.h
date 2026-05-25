#ifndef INFER_H
#define INFER_H

#include "config.h"
#include "memory.h"
#include "public.h"

#include <array>
#include <opencv2/opencv.hpp>
#include <vector>

struct Detection {
    // x1, y1, x2, y2
    std::array<float, 4> bbox;
    float                conf;
    int                  classId;
};

class YoloDetectModel {
  public:
    YoloDetectModel(const std::string trtFile,
                    int               raw_img_w,
                    int               raw_img_h,
                    int               gpuId      = GPU_ID,
                    float             nmsThresh  = NMS_THRESH,
                    float             confThresh = CONF_THRESH,
                    int               numClass   = NUM_CLASS);
    ~YoloDetectModel();
    std::vector<Detection> inference(const cv::Mat & img);
    void                   inferenceAsync(const cv::Mat & img);
    std::vector<Detection> postProcess(const float * output_data, const cv::Mat & img);

    void waitAsync();

    std::vector<Detection> getInferResultAsync(const cv::Mat & img);

  private:
    void getEngine();

  private:
    Logger      g_logger_;
    std::string trtFile_;

    int   numClass_;
    float nmsThresh_;
    float confThresh_;

    TrtEnginePtr  engine_;
    TrtRuntimePtr runtime_;
    TrtContextPtr context_;

    cudaStream_t stream_;

    std::vector<float>                   h_output_data_;
    std::array<unique_ptr_cuda<void>, 2> d_buffer_;
    unique_ptr_cuda<float>               d_transpose_;
    unique_ptr_cuda<float>               d_decode_;
    // preprocess
    unique_ptr_cuda<uchar>               d_src_data_;
    unique_ptr_cuda<uchar>               d_mid_data_;

    int  OUTPUT_CANDIDATES_;           // 8400: 80 * 80 + 40 * 40 + 20 * 20
    int  yolo26_max_num_output_bbox_;  // 暂时用于yolo26，后续可以删除
    int  yolo26_num_box_element_;      // 暂时用于yolo26，后续可以删除
    bool is_need_nms_ = true;

    int input_h_;
    int input_w_;
    int raw_img_h_;
    int raw_img_w_;
};

#endif  // INFER_H
