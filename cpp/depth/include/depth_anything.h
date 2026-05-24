#pragma once
#include "base.h"

#include <NvOnnxParser.h>

class DepthAnything : public BaseDepthModel {
  public:
    DepthAnything()           = default;
    ~DepthAnything() override = default;

    float              mean_[3] = { 0.485f, 0.456f, 0.406f };
    float              std_[3]  = { 0.229f, 0.224f, 0.225f };
    std::vector<float> preProcess(const cv::Mat & image) override;
};
