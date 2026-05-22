#pragma once
#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <tuple>

/**
 * @brief Depth_estimation structure
 */
struct DepthEstimation {
  int x;
  int y;
  int label;

  DepthEstimation() {
    x = 0;
    y = 0;
    label = -1;
  }
};

std::tuple<cv::Mat, int, int> resize_depth(cv::Mat &img, int w, int h);
