#include "lite_mono.h"

#include <opencv2/core/mat.hpp>

std::vector<float> LiteMono::Preprocess(const cv::Mat & image) {
    origin_img_h = image.rows;
    origin_img_w = image.cols;
    cv::Mat resized, rgb;
    cv::resize(image, resized, cv::Size(input_w, input_h), 0, 0, cv::INTER_LANCZOS4);
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    std::vector<float> input_tensor;
    input_tensor.reserve(3 * input_h * input_w);

    // Lite-Mono 仅做 /255.0，不减均值，CHW 格式
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < rgb.rows; ++i) {
            for (int j = 0; j < rgb.cols; ++j) {
                input_tensor.push_back((float) rgb.at<cv::Vec3b>(i, j)[c] / 255.0f);
            }
        }
    }
    return input_tensor;
}
