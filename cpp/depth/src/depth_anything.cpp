#include "depth_anything.h"

std::vector<float> DepthAnything::Preprocess(const cv::Mat & image) {
    origin_img_w = image.cols;
    origin_img_h = image.rows;
    // 自定义的 resize_depth (假设你在其他地方定义了它)
    cv::Mat resized_image, rgb;  // std::get<0>(resize_depth(image, input_w, input_h));
    cv::resize(image, resized_image, cv::Size(input_w, input_h));
    cv::cvtColor(resized_image, rgb, cv::COLOR_BGR2RGB);

    float mean[3] = { 0.485f, 0.456f, 0.406f };
    float std[3]  = { 0.229f, 0.224f, 0.225f };

    std::vector<float> input_tensor;
    for (int k = 0; k < 3; k++) {
        for (int i = 0; i < resized_image.rows; i++) {
            for (int j = 0; j < resized_image.cols; j++) {
                input_tensor.emplace_back(((float) rgb.at<cv::Vec3b>(i, j)[k] / 255.0f - mean[k]) / std[k]);
            }
        }
    }
    return input_tensor;
}
