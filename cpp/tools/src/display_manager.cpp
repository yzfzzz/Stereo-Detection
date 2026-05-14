#include "display_manager.h"

#include <experimental/filesystem>

// 全局鼠标回调函数
void onMouse(int event, int x, int y, int flags, void * userdata) {
    DisplayManager * dm = static_cast<DisplayManager *>(userdata);
    if (!dm || !dm->isEnabled()) {
        return;
    }

    if (event == cv::EVENT_LBUTTONDOWN) {
        dm->handleMouseClick(x, y);
    }
}

DisplayManager::DisplayManager(bool enabled, const std::string & window_name, cv::Size display_size) :
    enabled_(enabled),
    window_name_(window_name),
    display_size_(display_size) {
    if (enabled_) {
        cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
        cv::resizeWindow(window_name_, display_size_.width, display_size_.height);
        cv::setMouseCallback(window_name_, onMouse, this);
        std::cout << "DisplayManager initialized" << std::endl;
    }
}

DisplayManager::~DisplayManager() {
    if (enabled_) {
        cv::destroyWindow(window_name_);
    }
}

float DisplayManager::computeMeanDepth(const std::vector<float> & tlwh) const {
    // 调试：检查深度图状态
    if (depth_map_.empty()) {
        std::cout << "[DEBUG] depth_map is empty!" << std::endl;
        return 0.0f;
    }

    std::cout << "[DEBUG] depth_map: size=" << depth_map_.size() << ", type=" << depth_map_.type()
              << ", channels=" << depth_map_.channels() << std::endl;

    const int num_samples = 64;  // 采样点数（5x5网格）
    float     sum_depth   = 0.0f;
    int       valid_count = 0;
    int       zero_count  = 0;

    // BBox 边界
    int left   = static_cast<int>(tlwh[0]);
    int top    = static_cast<int>(tlwh[1]);
    int right  = static_cast<int>(tlwh[0] + tlwh[2]);
    int bottom = static_cast<int>(tlwh[1] + tlwh[3]);

    // 确保边界在图像范围内
    left   = std::max(0, std::min(left, depth_map_.cols - 1));
    top    = std::max(0, std::min(top, depth_map_.rows - 1));
    right  = std::max(0, std::min(right, depth_map_.cols - 1));
    bottom = std::max(0, std::min(bottom, depth_map_.rows - 1));

    // 计算采样步长
    int width  = right - left;
    int height = bottom - top;

    if (width <= 0 || height <= 0) {
        return 0.0f;
    }

    // 在 BBox 内均匀采样（5x5 网格）
    int   grid_size = static_cast<int>(std::sqrt(num_samples));
    float step_x    = static_cast<float>(width) / (grid_size - 1);
    float step_y    = static_cast<float>(height) / (grid_size - 1);

    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            int x = left + static_cast<int>(i * step_x);
            int y = top + static_cast<int>(j * step_y);

            // 边界检查
            if (x < 0 || x >= depth_map_.cols || y < 0 || y >= depth_map_.rows) {
                continue;
            }

            float depth = 0.0f;
            if (depth_map_.type() == CV_32FC1) {
                depth = depth_map_.at<float>(y, x);
            } else if (depth_map_.type() == CV_8UC1) {
                depth = static_cast<float>(depth_map_.at<uchar>(y, x));
            } else {
                // 处理其他类型
                std::cout << "[DEBUG] Unsupported depth map type: " << depth_map_.type() << std::endl;
            }

            if (depth == 0) {
                zero_count++;
            } else if (depth > 0) {
                sum_depth += depth;
                valid_count++;
            }
        }
    }

    std::cout << "[DEBUG] Samples: total=" << num_samples << ", valid=" << valid_count << ", zero=" << zero_count
              << std::endl;

    return (valid_count > 0) ? (sum_depth / valid_count) : 0.0f;
}

void DisplayManager::printTargetInfo(const STrack & track) const {
    int                        class_id = track.class_id;
    int                        track_id = track.track_id;
    const std::vector<float> & tlwh     = track.tlwh;

    // 使用多点采样计算深度均值
    float depth = computeMeanDepth(tlwh);

    std::cout << "\n=== Target Info ===" << std::endl;
    std::cout << "Class: " << vClassNames[class_id] << std::endl;
    std::cout << "Track ID: " << track_id << std::endl;
    std::cout << "Depth (mean of 25 samples): " << depth << std::endl;
    std::cout << "BBox: [" << tlwh[0] << ", " << tlwh[1] << ", " << tlwh[0] + tlwh[2] << ", " << tlwh[1] + tlwh[3]
              << "]" << std::endl;
    std::cout << "==================\n" << std::endl;
}

void DisplayManager::handleMouseClick(int x, int y) {
    for (const auto & track : tracks_) {
        const std::vector<float> & tlwh   = track.tlwh;
        float                      left   = tlwh[0];
        float                      top    = tlwh[1];
        float                      right  = tlwh[0] + tlwh[2];
        float                      bottom = tlwh[1] + tlwh[3];

        if (x >= left && x <= right && y >= top && y <= bottom) {
            printTargetInfo(track);
            return;
        }
    }
    std::cout << "No target clicked" << std::endl;
}

void DisplayManager::updateData(const std::vector<STrack> & tracks, const cv::Mat & depth_map) {
    if (enabled_) {
        tracks_ = tracks;
        if (!depth_map.empty()) {
            depth_map_ = depth_map.clone();
            cv::resize(depth_map_, depth_map_, cv::Size(1280, 720));
        }
    }
}

int DisplayManager::handleKey(int key) {
    if (!enabled_) {
        return key;  // 返回原始按键
    }

    if (key == Key_Input::SPACE) {
        std::cout << "Paused, press SPACE to continue..." << std::endl;
        while (true) {
            char pause_key = cv::waitKey(0);
            if (pause_key == Key_Input::SPACE) {
                std::cout << "Resuming..." << std::endl;
                break;
            } else if (pause_key == Key_Input::ESC) {  // ESC键退出
                std::cout << "User exited" << std::endl;
                return Key_Input::ESC;
            }
        }
        return 0;
    } else if (key == Key_Input::ESC) {  // ESC键退出
        std::cout << "User exited" << std::endl;
        return Key_Input::ESC;
    }
    return key;
}

void DisplayManager::show(const cv::Mat & frame) {
    if (enabled_) {
        cv::imshow(window_name_, frame);
    }
}

int DisplayManager::waitKey(int delay) {
    if (!enabled_) {
        return Key_Input::ESC;
    }
    return cv::waitKey(delay);
}

DepthPlotter::DepthPlotter(const std::string & out_dir) : out_dir_(out_dir) {
    if (!std::experimental::filesystem::exists(out_dir_)) {
        std::experimental::filesystem::create_directories(out_dir_);
    }
}

DepthPlotter::~DepthPlotter() {}

void DepthPlotter::update(int track_id, int frame_id, float current_depth) {
    if (current_depth > 0) {
        track_depth_history_[track_id].push_back({ frame_id, current_depth });
    }
}

void DepthPlotter::drawSinglePlot(cv::Mat &                                  canvas,
                                  const cv::Rect &                           roi,
                                  int                                        track_id,
                                  const std::vector<std::pair<int, float>> & history) const {
    if (history.empty()) {
        return;
    }

    // 背景和边框
    cv::rectangle(canvas, roi, cv::Scalar(240, 240, 240), -1);
    cv::rectangle(canvas, roi, cv::Scalar(0, 0, 0), 2);

    int min_x = history.front().first;
    int max_x = history.back().first;
    if (min_x == max_x) {
        max_x = min_x + 1;  // 避免除以 0
    }

    float min_y = history[0].second, max_y = history[0].second;
    for (const auto & pt : history) {
        min_y = std::min(min_y, pt.second);
        max_y = std::max(max_y, pt.second);
    }

    // Y 轴适当留白
    float pad_y = std::max(0.5f, (max_y - min_y) * 0.15f);
    min_y       = std::max(0.0f, min_y - pad_y);
    max_y       = max_y + pad_y;

    int pad_left = 60, pad_bottom = 30;
    int plot_w = roi.width - pad_left - 15;
    int plot_h = roi.height - pad_bottom - 45;

    int plot_x0 = roi.x + pad_left;
    int plot_y0 = roi.y + roi.height - pad_bottom;

    // 绘制标签
    cv::putText(canvas, "Track ID: " + std::to_string(track_id), cv::Point(roi.x + 10, roi.y + 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

    cv::putText(canvas, cv::format("%.2f", max_y), cv::Point(roi.x + 5, roi.y + 50), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                cv::Scalar(80, 80, 80));
    cv::putText(canvas, cv::format("%.2f", min_y), cv::Point(roi.x + 5, plot_y0 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                cv::Scalar(80, 80, 80));

    // 绘制折线
    for (size_t i = 1; i < history.size(); ++i) {
        int x1 = plot_x0 + (history[i - 1].first - min_x) * plot_w / (max_x - min_x);
        int y1 = plot_y0 - (history[i - 1].second - min_y) * plot_h / (max_y - min_y);

        int x2 = plot_x0 + (history[i].first - min_x) * plot_w / (max_x - min_x);
        int y2 = plot_y0 - (history[i].second - min_y) * plot_h / (max_y - min_y);

        cv::line(canvas, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
        cv::circle(canvas, cv::Point(x2, y2), 3, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
    }
}

void DepthPlotter::savePlots() {
    if (track_depth_history_.empty()) {
        return;
    }

    const int canvas_w         = 1920;
    const int canvas_h         = 1080;
    const int grid_cols        = 4;
    const int grid_rows        = 4;
    const int plots_per_canvas = grid_cols * grid_rows;  // 16

    int cell_w = canvas_w / grid_cols;
    int cell_h = canvas_h / grid_rows;

    std::vector<int> valid_tracks;
    for (const auto & pair : track_depth_history_) {
        if (pair.second.size() >= 3) {  // 只画那些存在超过 3 帧数据的
            valid_tracks.push_back(pair.first);
        }
    }

    int total_tracks = valid_tracks.size();
    if (total_tracks == 0) {
        return;
    }

    int canvas_count = (total_tracks + plots_per_canvas - 1) / plots_per_canvas;

    for (int c = 0; c < canvas_count; ++c) {
        cv::Mat canvas(canvas_h, canvas_w, CV_8UC3, cv::Scalar(255, 255, 255));

        for (int i = 0; i < plots_per_canvas; ++i) {
            int track_idx = c * plots_per_canvas + i;
            if (track_idx >= total_tracks) {
                break;
            }

            int      track_id = valid_tracks[track_idx];
            int      row      = i / grid_cols;
            int      col      = i % grid_cols;
            cv::Rect roi(col * cell_w, row * cell_h, cell_w, cell_h);

            drawSinglePlot(canvas, roi, track_id, track_depth_history_[track_id]);
        }

        std::string filename = out_dir_ + "/depth_trend_canvas_" + std::to_string(c + 1) + ".jpg";
        cv::imwrite(filename, canvas);
        std::cout << "[DepthPlotter] Saved: " << filename << std::endl;
    }
}
