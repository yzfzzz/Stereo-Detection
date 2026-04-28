#ifndef INFER_H
#define INFER_H

#include "config.h"
#include "public.h"
#include "types.h"
#include <opencv2/opencv.hpp>

using namespace nvinfer1;

class YoloDetector {
public:
  YoloDetector(const std::string trtFile, int gpuId = kGpuId,
               float nmsThresh = kNmsThresh, float confThresh = kConfThresh,
               int numClass = kNumClass);
  ~YoloDetector();
  std::vector<Detection> inference(cv::Mat &img);
  static void draw_image(cv::Mat &img, std::vector<Detection> &inferResult);

private:
  void get_engine();

private:
  Logger gLogger;
  std::string trtFile_;

  int numClass_;
  float nmsThresh_;
  float confThresh_;

  ICudaEngine *engine;
  IRuntime *runtime;
  IExecutionContext *context;

  cudaStream_t stream;

  float *outputData;
  std::vector<void *> vBufferD;
  float *transposeDevice;
  float *decodeDevice;

  int OUTPUT_CANDIDATES; // 8400: 80 * 80 + 40 * 40 + 20 * 20
  int yolo26_max_num_output_bbox; // 暂时用于yolo26，后续可以删除
  int yolo26_num_box_element; // 暂时用于yolo26，后续可以删除
  bool is_need_nms_ = true;
};

#endif // INFER_H
