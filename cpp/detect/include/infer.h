#ifndef INFER_H
#define INFER_H

#include "config.h"
#include "public.h"
#include "types.h"

#include <opencv2/opencv.hpp>

class YoloDetector {
  public:
    YoloDetector(const std::string trtFile,
                 int               raw_img_w,
                 int               raw_img_h,
                 int               gpuId      = GPU_ID,
                 float             nmsThresh  = NMS_THRESH,
                 float             confThresh = CONF_THRESH,
                 int               numClass   = NUM_CLASS);
    ~YoloDetector();
    std::vector<Detection> inference(const cv::Mat & img);
    void                   inferenceAsync(const cv::Mat & img);
    std::vector<Detection> postProcess(float * output_data, const cv::Mat & img);
    static void            drawImage(cv::Mat & img, std::vector<Detection> & infer_result);

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

    nvinfer1::ICudaEngine *       engine_;
    nvinfer1::IRuntime *          runtime_;
    nvinfer1::IExecutionContext * context_;

    cudaStream_t stream_;

    float *             output_data_;
    std::vector<void *> v_buffer_d_;
    float *             transpose_device_;
    float *             decode_device_;
    // preprocess
    uchar *             src_dev_data_;
    uchar *             mid_dev_data_;

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
