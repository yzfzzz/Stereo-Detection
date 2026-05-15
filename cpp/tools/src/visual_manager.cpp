#include "visual_manager.h"
#include "config.h"
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

void DepthPlotter::update(int track_id, int frame_id, float current_depth, float velocity) {
    if (current_depth > 0) {
        // 保存三大信息：帧号，深度，速度
        track_depth_history_[track_id].push_back({ frame_id, current_depth, velocity });
    }
}

void DepthPlotter::drawSinglePlot(cv::Mat &                                          canvas,
                                  const cv::Rect &                                   roi,
                                  int                                                track_id,
                                  const std::vector<std::tuple<int, float, float>> & history) const {
    if (history.empty()) {
        return;
    }

    // ========== 1. 背景和边框 ==========
    cv::rectangle(canvas, roi, cv::Scalar(240, 240, 240), -1);
    cv::rectangle(canvas, roi, cv::Scalar(0, 0, 0), 2);

    int min_x = std::get<0>(history.front());
    int max_x = std::get<0>(history.back());
    if (min_x == max_x) {
        max_x = min_x + 1;  // 避免除以 0
    }

    // ========== 2. 计算 Y 轴极值 ==========
    float min_depth = std::get<1>(history[0]), max_depth = std::get<1>(history[0]);
    float min_vel = std::get<2>(history[0]), max_vel = std::get<2>(history[0]);

    for (const auto & pt : history) {
        min_depth = std::min(min_depth, std::get<1>(pt));
        max_depth = std::max(max_depth, std::get<1>(pt));
        min_vel   = std::min(min_vel, std::get<2>(pt));
        max_vel   = std::max(max_vel, std::get<2>(pt));
    }

    // Y轴留白
    float pad_depth = std::max(0.5f, (max_depth - min_depth) * 0.15f);
    min_depth       = std::max(0.0f, min_depth - pad_depth);
    max_depth       = max_depth + pad_depth;

    float pad_vel = std::max(0.1f, (max_vel - min_vel) * 0.15f);
    min_vel       = min_vel - pad_vel;
    max_vel       = max_vel + pad_vel;

    if (max_vel - min_vel < 0.01f) {
        max_vel += 0.05f;
        min_vel -= 0.05f;
    }

    // ========== 3. 上下分屏设计 ==========
    int pad_left = 60, pad_bottom = 20, pad_top = 30;
    int plot_w = roi.width - pad_left - 15;

    int total_plot_h = roi.height - pad_top - pad_bottom;
    int sub_plot_h   = (total_plot_h - 20) / 2;  // 两个图之间留出 20px 间隙

    // 坐标系原点(左下角起点)
    int depth_x0 = roi.x + pad_left;
    int depth_y0 = roi.y + pad_top + sub_plot_h;

    int vel_x0 = roi.x + pad_left;
    int vel_y0 = roi.y + roi.height - pad_bottom;

    // ========== 4. 绘制文字标签 ==========
    cv::putText(canvas, "Track ID: " + std::to_string(track_id), cv::Point(roi.x + 5, roi.y + 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);

    // Depth 刻度 (上半)
    cv::putText(canvas, cv::format("D:%.2f", max_depth), cv::Point(roi.x + 2, roi.y + pad_top + 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(100, 100, 100));
    cv::putText(canvas, cv::format("%.2f", min_depth), cv::Point(roi.x + 5, depth_y0), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                cv::Scalar(100, 100, 100));

    // Vel 刻度 (下半)
    cv::putText(canvas, cv::format("V:%.2f", max_vel), cv::Point(roi.x + 2, depth_y0 + 30), cv::FONT_HERSHEY_SIMPLEX,
                0.4, cv::Scalar(100, 100, 100));
    cv::putText(canvas, cv::format("%.2f", min_vel), cv::Point(roi.x + 5, vel_y0), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                cv::Scalar(100, 100, 100));

    if (min_vel < 0 && max_vel > 0) {
        int zero_y = vel_y0 - (0.0f - min_vel) * sub_plot_h / (max_vel - min_vel);
        cv::line(canvas, cv::Point(vel_x0, zero_y), cv::Point(vel_x0 + plot_w, zero_y), cv::Scalar(150, 150, 150), 1,
                 cv::LINE_AA);
    }

    // ========== 5. 绘制折线 ==========
    for (size_t i = 1; i < history.size(); ++i) {
        // ------ 画深度图 ------
        int x1_d = depth_x0 + (std::get<0>(history[i - 1]) - min_x) * plot_w / (max_x - min_x);
        int y1_d = depth_y0 - (std::get<1>(history[i - 1]) - min_depth) * sub_plot_h / (max_depth - min_depth);
        int x2_d = depth_x0 + (std::get<0>(history[i]) - min_x) * plot_w / (max_x - min_x);
        int y2_d = depth_y0 - (std::get<1>(history[i]) - min_depth) * sub_plot_h / (max_depth - min_depth);

        cv::line(canvas, cv::Point(x1_d, y1_d), cv::Point(x2_d, y2_d), cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
        cv::circle(canvas, cv::Point(x2_d, y2_d), 2, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);

        // ------ 画速度图 ------
        int x1_v = vel_x0 + (std::get<0>(history[i - 1]) - min_x) * plot_w / (max_x - min_x);
        int y1_v = vel_y0 - (std::get<2>(history[i - 1]) - min_vel) * sub_plot_h / (max_vel - min_vel);
        int x2_v = vel_x0 + (std::get<0>(history[i]) - min_x) * plot_w / (max_x - min_x);
        int y2_v = vel_y0 - (std::get<2>(history[i]) - min_vel) * sub_plot_h / (max_vel - min_vel);

        cv::line(canvas, cv::Point(x1_v, y1_v), cv::Point(x2_v, y2_v), cv::Scalar(0, 200, 0), 2, cv::LINE_AA);
        cv::circle(canvas, cv::Point(x2_v, y2_v), 2, cv::Scalar(200, 0, 200), -1, cv::LINE_AA);
    }
}

void DepthPlotter::savePlots() {
    if (track_depth_history_.empty()) {
        return;
    }

    const int canvas_w         = 3840;
    const int canvas_h         = 2160;
    const int grid_cols        = 8;
    const int grid_rows        = 4;
    const int plots_per_canvas = grid_cols * grid_rows;  // 64

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

DrawingManager::DrawingManager(const std::vector<std::string> & class_names) : vClassNames_(class_names) {}

void DrawingManager::drawTrackedObject(cv::Mat &                     img,
                                       const STrack &                track,
                                       const MotionStateInfoRecord & motion_state,
                                       cv::Scalar                    color) {
    const std::vector<float> & tlwh     = track.tlwh;
    int                        class_id = track.class_id;
    int                        track_id = track.track_id;

    // 1. 获取运动状态字符串
    auto        it               = MOTION_STR_MAP.find({ motion_state.state_vec, motion_state.state_acc });
    std::string motion_state_str = (it != MOTION_STR_MAP.end()) ? it->second : "Unknown";

    // 2. 准备文字标签
    std::string label = cv::format("%s #%d [%s]", vClassNames_[class_id].c_str(), track_id, motion_state_str.c_str());

    // 3. 绘制文字背景框和文字
    int      baseLine   = 0;
    cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseLine);
    cv::Rect rect_bg(cv::Point((int) tlwh[0], (int) tlwh[1] - label_size.height - 8),
                     cv::Size(label_size.width + 8, label_size.height + 8));

    cv::rectangle(img, rect_bg, color, cv::FILLED);
    cv::putText(img, label, cv::Point((int) tlwh[0] + 4, (int) tlwh[1] - 4), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

    // 4. 绘制目标主体矩形框
    cv::rectangle(img, cv::Rect((int) tlwh[0], (int) tlwh[1], (int) tlwh[2], (int) tlwh[3]), color, 2);

    // 5. 检查运动状态是否为"加速靠近"，如果是则绘制红色交叉
    if (motion_state.state_vec == MotionState::APPROACH && motion_state.state_acc == MotionState::ACCELE) {
        int x1 = static_cast<int>(tlwh[0]);
        int y1 = static_cast<int>(tlwh[1]);
        int x2 = static_cast<int>(tlwh[0] + tlwh[2]);
        int y2 = static_cast<int>(tlwh[1] + tlwh[3]);

        cv::Scalar red_color(0, 0, 255);  // BGR 格式的红色
        int        line_thickness = 2;

        // 绘制左上到右下的线
        cv::line(img, cv::Point(x1, y1), cv::Point(x2, y2), red_color, line_thickness, cv::LINE_AA);
        // 绘制左下到右上的线
        cv::line(img, cv::Point(x1, y2), cv::Point(x2, y1), red_color, line_thickness, cv::LINE_AA);
    }
}

void DrawingManager::drawGlobalInfo(cv::Mat & img, int num_frames, int show_fps, size_t num_tracks) {
    cv::putText(img, cv::format("frame: %d fps: %d num: %zu", num_frames, show_fps, num_tracks), cv::Point(0, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
}

cv::Mat DrawingManager::concatenateFrames(const cv::Mat & rgb_img, const cv::Mat & depth_vis) {
    cv::Mat out_frame;

    // 如果没有深度图，直接返回原图的一份拷贝
    if (depth_vis.empty()) {
        rgb_img.copyTo(out_frame);
        return out_frame;
    }

    // 确保深度可视化的宽度高度跟原图一致，避免拼接越界崩溃
    cv::Mat resized_depth_vis;
    if (depth_vis.size() != rgb_img.size()) {
        cv::resize(depth_vis, resized_depth_vis, rgb_img.size());
    } else {
        resized_depth_vis = depth_vis;
    }

    // 创建 (原图高度 + 深度图高度) 作为总高度，宽度不变
    out_frame.create(rgb_img.rows + resized_depth_vis.rows, rgb_img.cols, rgb_img.type());

    // 拷贝原图到上半部分
    rgb_img.copyTo(out_frame(cv::Rect(0, 0, rgb_img.cols, rgb_img.rows)));

    // 拷贝深度图到下半部分
    resized_depth_vis.copyTo(out_frame(cv::Rect(0, rgb_img.rows, rgb_img.cols, resized_depth_vis.rows)));

    return out_frame;
}
