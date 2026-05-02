#pragma once
#include "base.h"

class LiteMono : public BaseDepthModel {
  public:
    LiteMono()           = default;
    ~LiteMono() override = default;

  protected:
    std::vector<float> Preprocess(const cv::Mat & image) override;
};
