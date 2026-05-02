#pragma once
#include "base.h"

#include <NvOnnxParser.h>

class DepthAnything : public BaseDepthModel {
  public:
    DepthAnything()           = default;
    ~DepthAnything() override = default;


  protected:
    std::vector<float> Preprocess(const cv::Mat & image) override;
};
