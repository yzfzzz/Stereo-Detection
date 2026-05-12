#include "display_manager.h"

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

DisplayManager::DisplayManager(bool enabled, const std::string & window_name) :
    enabled_(enabled),
    window_name_(window_name) {
    if (enabled_) {
        cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
        cv::resizeWindow(window_name_, 1280, 720);
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
