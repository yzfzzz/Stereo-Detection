#include "depth_anything.h"

std::vector<float> DepthAnything::preProcess(const cv::Mat & image) {
    origin_img_w_ = image.cols;
    origin_img_h_ = image.rows;
    // 自定义的 resize_depth (假设你在其他地方定义了它)
    cv::Mat resized_image, rgb;  // std::get<0>(resize_depth(image, input_w, input_h));
    cv::resize(image, resized_image, cv::Size(input_w_, input_h_));
    cv::cvtColor(resized_image, rgb, cv::COLOR_BGR2RGB);

    std::vector<float> input_tensor;

    for (int k = 0; k < 3; k++) {
        for (int i = 0; i < resized_image.rows; i++) {
            for (int j = 0; j < resized_image.cols; j++) {
                input_tensor.emplace_back(((float) rgb.at<cv::Vec3b>(i, j)[k] / 255.0f - mean_[k]) / std_[k]);
            }
        }
    }
    return input_tensor;
}
