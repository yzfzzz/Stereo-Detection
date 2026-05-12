#pragma once
#include "config.h"
#include "STrack.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

// 运动状态引擎
class MotionStateEngine {
  public:
    MotionStateEngine();
    ~MotionStateEngine();

    // 根据深度和时间戳计算运动状态
    std::string computeMotionState(float depth, double timestamp);

    float getObjectDepth(cv::Mat depth, const STrack & track, cv::Size image_size) {
        if (!depth.empty()) {
            cv::resize(depth, depth, image_size);
        } else {
            std::cerr << "[DEBUG] depth_map is empty!" << std::endl;
            return 0.0f;
        }

        const std::vector<float> & tlwh        = track.tlwh;
        float                      left        = tlwh[0];
        float                      top         = tlwh[1];
        float                      right       = tlwh[0] + tlwh[2];
        float                      bottom      = tlwh[1] + tlwh[3];
        float                      depth_value = 0.0f;

        int class_id = track.class_id;
        int track_id = track.track_id;
        // 使用多点采样计算深度均值
        depth_value  = computeMeanDepth(depth, tlwh);

        printf(
            "\n=== Target Info ===\n"
            "Class: %s\n"
            "Track ID: %d\n"
            "Depth (mean of 25 samples): %f\n"
            "BBox: [%f, %f, %f, %f]\n"
            "==================\n\n",
            vClassNames[class_id].c_str(), track_id, depth_value, tlwh[0], tlwh[1], tlwh[0] + tlwh[2],
            tlwh[1] + tlwh[3]);

        return depth_value;
    }

    float computeMeanDepth(cv::Mat depth, const std::vector<float> & tlwh, int num_samples = 25) const {
        printf("[DEBUG] depth_map: size=(%d, %d), type=%d, channels=%d\n", depth.cols, depth.rows, depth.type(),
               depth.channels());

        if (sqrt(num_samples) != static_cast<int>(sqrt(num_samples)) || num_samples <= 1) {
            std::cerr << "[DEBUG] num_samples should be a perfect square for grid sampling." << std::endl;
            return 0.0f;
        }

        float sum_depth   = 0.0f;
        int   valid_count = 0;
        int   zero_count  = 0;

        // BBox 边界
        int left   = static_cast<int>(tlwh[0]);
        int top    = static_cast<int>(tlwh[1]);
        int right  = static_cast<int>(tlwh[0] + tlwh[2]);
        int bottom = static_cast<int>(tlwh[1] + tlwh[3]);

        // 确保边界在图像范围内
        left   = std::max(0, std::min(left, depth.cols - 1));
        top    = std::max(0, std::min(top, depth.rows - 1));
        right  = std::max(0, std::min(right, depth.cols - 1));
        bottom = std::max(0, std::min(bottom, depth.rows - 1));

        // 计算采样步长
        int width  = right - left;
        int height = bottom - top;

        if (width <= 0 || height <= 0) {
            return 0.0f;
        }

        // 在 BBox 内均匀采样（grid_size x grid_size 网格）
        int                           grid_size = static_cast<int>(std::sqrt(num_samples));
        std::vector<std::vector<int>> weight_matrix(grid_size, std::vector<int>(grid_size, 0));
        int                           weight_sum = 0;
        for (int i = 1; i <= (grid_size + 1) / 2; i++) {
            for (int j = 1; j <= (grid_size + 1) / 2; j++) {
                weight_matrix[i - 1][j - 1]                 = i * j;
                weight_matrix[grid_size - i][j - 1]         = i * j;
                weight_matrix[i - 1][grid_size - j]         = i * j;
                weight_matrix[grid_size - i][grid_size - j] = i * j;
            }
        }

        float step_x = static_cast<float>(width) / (grid_size - 1);
        float step_y = static_cast<float>(height) / (grid_size - 1);

        for (int i = 0; i < grid_size; ++i) {
            for (int j = 0; j < grid_size; ++j) {
                int x = left + static_cast<int>(i * step_x);
                int y = top + static_cast<int>(j * step_y);

                // 边界检查
                if (x < 0 || x >= depth.cols || y < 0 || y >= depth.rows) {
                    continue;
                }

                float depth_value = 0.0f;
                if (depth.type() == CV_32FC1) {
                    depth_value = depth.at<float>(y, x);
                } else if (depth.type() == CV_8UC1) {
                    depth_value = static_cast<float>(depth.at<uchar>(y, x));
                } else {
                    // 处理其他类型
                    std::cout << "[DEBUG] Unsupported depth map type: " << depth.type() << std::endl;
                }

                if (depth_value == 0) {
                    zero_count++;
                } else if (depth_value > 0) {
                    sum_depth += depth_value * weight_matrix[i][j];
                    weight_sum += weight_matrix[i][j];
                    valid_count++;
                }
            }
        }

        std::cout << "[DEBUG] Samples: total=" << num_samples << ", valid=" << valid_count << ", zero=" << zero_count
                  << std::endl;

        return (valid_count > 0) ? (sum_depth / (valid_count * weight_sum)) : 0.0f;
    }

  private:
};
