#include "motion_state_engine.h"

#include <iostream>
#include <opencv2/core/operations.hpp>
#include <utility>

MotionStateEngine::MotionStateEngine(int   sma_window_size,
                                     float velocity_threshold,
                                     float acceleration_threshold,
                                     float jump_threshold,
                                     int   consistency_frames) :
    sma_window_size_(sma_window_size),
    velocity_threshold_(velocity_threshold),
    acceleration_threshold_(acceleration_threshold),
    jump_threshold_(jump_threshold),
    consistency_frames_(consistency_frames) {
    printf("[MotionStateEngine] SMA window=%d, vel_thresh=%.2f, acc_thresh=%.2f, jump_thresh=%.2f, consistency=%d\n",
           sma_window_size_, velocity_threshold_, acceleration_threshold_, jump_threshold_, consistency_frames_);
}

bool MotionStateEngine::isOutlier(int track_id, float raw_depth) const {
    auto it = depth_history_queues_.find(track_id);
    if (it == depth_history_queues_.end() || it->second.empty()) {
        return false;
    }

    float median = computeMedian(it->second);
    if (median <= 0.0f) {
        return false;
    }

    float relative_change = std::abs(raw_depth - median) / median;
    return relative_change > jump_threshold_;
}

float MotionStateEngine::computeMedian(const std::deque<float> & queue) const {
    if (queue.empty()) {
        return 0.0f;
    }

    std::vector<float> sorted(queue.begin(), queue.end());
    std::sort(sorted.begin(), sorted.end());

    size_t n = sorted.size();
    if (n % 2 == 0) {
        return (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0f;
    } else {
        return sorted[n / 2];
    }
}

bool MotionStateEngine::checkConsistency(int track_id, MotionState direction) const {
    auto it = direction_history_.find(track_id);
    if (it == direction_history_.end() || it->second.size() < static_cast<size_t>(consistency_frames_)) {
        return false;
    }

    int count = 0;
    for (auto rit = it->second.rbegin(); rit != it->second.rend() && count < consistency_frames_; ++rit, ++count) {
        if (*rit != direction) {
            return false;
        }
    }
    return true;
}

std::pair<MotionState, MotionState> MotionStateEngine::computeMotionState(int    track_id,
                                                                          float  raw_depth,
                                                                          double timestamp) {
    if (raw_depth <= 0.0f) {
        return { INVAILD, INVAILD };
    }

    auto & history = history_states_[track_id];
    size_t window_size = static_cast<size_t>(consistency_frames_) + 3;  // 适当扩大窗口能拟合得更准，比如5~8帧

    // 直接存入原始深度到 smoothed_depth 字段中复用结构体
    history.push_back({ raw_depth, timestamp, 0.0f });
    if (history.size() > window_size) {
        history.pop_front();
    }

    if (history.size() < 3) {  // 点数太少无法有效拟合
        return { STABLE, CONSTANT };
    }

    // ========== 1. 最小二乘法计算趋势斜率 (拟合速度) ==========
    int    n     = history.size();
    double sum_t = 0.0, sum_d = 0.0, sum_td = 0.0, sum_tt = 0.0;

    // 为了防止 timestamp 过大导致浮点精度丢失，以第一帧时间为基准点 (t = 0)
    double t0 = history.front().timestamp;

    for (const auto & state : history) {
        double t = state.timestamp - t0;
        double d = state.smoothed_depth;
        sum_t += t;
        sum_d += d;
        sum_td += t * d;
        sum_tt += t * t;
    }

    // 直线斜率公式: m = (n*Σ(xy) - Σx*Σy) / (n*Σ(x^2) - (Σx)^2)
    // 这里的斜率 slope 相当于一段时间内的稳定速度
    double denominator    = (n * sum_tt - sum_t * sum_t);
    float  trend_velocity = 0.0f;
    if (std::abs(denominator) > 1e-6) {
        trend_velocity = static_cast<float>((n * sum_td - sum_t * sum_d) / denominator);
    }

    // 更新当前速度参数给外部或者加速度计算使用
    history.back().velocity = trend_velocity;

    // ========== 2. 判定方向 ==========
    MotionState direction_state = STABLE;
    if (trend_velocity > velocity_threshold_) {
        direction_state = MOVE_AWAY;
    } else if (trend_velocity < -velocity_threshold_) {
        direction_state = APPROACH;
    }

    // ========== 3. 计算加速度 (可选项，可基于近几帧或直接默认) ==========
    MotionState accel_state = CONSTANT;
    // 省略复杂求导，或者使用 trend_velocity 的一阶差分计算加速度...

    return { direction_state, accel_state };
}

float MotionStateEngine::getObjectDepth(cv::Mat depth, const STrack & track, cv::Size image_size) {
    if (!depth.empty()) {
        cv::resize(depth, depth, image_size);
    } else {
        std::cerr << "[DEBUG] depth_map is empty!" << std::endl;
        return 0.0f;
    }

    const std::vector<float> & tlwh        = track.tlwh;
    float                      depth_value = 0.0f;

    depth_value = computeMeanDepth(depth, tlwh);

    return depth_value;
}

float MotionStateEngine::computeMeanDepth(cv::Mat depth, const std::vector<float> & tlwh, int num_samples) const {
    if (sqrt(num_samples) != static_cast<int>(sqrt(num_samples)) || num_samples < 1) {
        std::cerr << "[DEBUG] num_samples should be a perfect square for grid sampling." << std::endl;
        return 0.0f;
    }

    float sum_depth  = 0.0f;
    int   zero_count = 0;

    int left   = static_cast<int>(tlwh[0]);
    int top    = static_cast<int>(tlwh[1]);
    int right  = static_cast<int>(tlwh[0] + tlwh[2]);
    int bottom = static_cast<int>(tlwh[1] + tlwh[3]);

    left   = std::max(0, std::min(left, depth.cols - 1));
    top    = std::max(0, std::min(top, depth.rows - 1));
    right  = std::max(0, std::min(right, depth.cols - 1));
    bottom = std::max(0, std::min(bottom, depth.rows - 1));

    int x = static_cast<int>((left + right) / 2);
    int y = static_cast<int>((top + bottom) / 2);

    float depth_value = 0.0f;
    if (depth.type() == CV_32FC1) {
        depth_value = depth.at<float>(y, x);
    } else if (depth.type() == CV_8UC1) {
        depth_value = static_cast<float>(depth.at<uchar>(y, x));
    }
    return depth_value;

    // int width  = right - left;
    // int height = bottom - top;

    // if (width <= 0 || height <= 0) {
    //     return 0.0f;
    // }

    // int                           grid_size = static_cast<int>(std::sqrt(num_samples));
    // std::vector<std::vector<int>> weight_matrix(grid_size, std::vector<int>(grid_size, 0));
    // int                           weight_sum = 0;
    // for (int i = 1; i <= (grid_size + 1) / 2; i++) {
    //     for (int j = 1; j <= (grid_size + 1) / 2; j++) {
    //         weight_matrix[i - 1][j - 1]                 = i * j;
    //         weight_matrix[grid_size - i][j - 1]         = i * j;
    //         weight_matrix[i - 1][grid_size - j]         = i * j;
    //         weight_matrix[grid_size - i][grid_size - j] = i * j;
    //     }
    // }

    // float step_x = static_cast<float>(width) / (grid_size - 1);
    // float step_y = static_cast<float>(height) / (grid_size - 1);

    // for (int i = 0; i < grid_size; ++i) {
    //     for (int j = 0; j < grid_size; ++j) {
    //         int x = left + static_cast<int>((i + 0.5f) * step_x);
    //         int y = top + static_cast<int>((j + 0.5f) * step_y);

    //         if (x < 0 || x >= depth.cols || y < 0 || y >= depth.rows) {
    //             continue;
    //         }

    //         float depth_value = 0.0f;
    //         if (depth.type() == CV_32FC1) {
    //             depth_value = depth.at<float>(y, x);
    //         } else if (depth.type() == CV_8UC1) {
    //             depth_value = static_cast<float>(depth.at<uchar>(y, x));
    //         }

    //         if (depth_value == 0) {
    //             zero_count++;
    //         } else if (depth_value > 0) {
    //             sum_depth += depth_value * static_cast<float>(weight_matrix[i][j]);
    //             weight_sum += weight_matrix[i][j];
    //         }
    //     }
    // }

    // return (zero_count < num_samples) ? (sum_depth / weight_sum) : 0.0f;
}
