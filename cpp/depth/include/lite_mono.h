#pragma once
#include "base.h"

class LiteMono : public BaseDepthModel {
  public:
    LiteMono()           = default;
    ~LiteMono() override = default;

    std::vector<float> preProcess(const cv::Mat & image) override;
};
