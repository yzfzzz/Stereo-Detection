#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// 前向声明
class STrack;
class DisplayManager;

// 鼠标回调函数前向声明
void onMouse(int event, int x, int y, int flags, void* userdata);

// 全局变量声明（来自 config.h）
extern const std::vector<std::string> vClassNames;

// 显示管理器类：负责窗口管理、显示、鼠标点击等事件
class DisplayManager {
  private:
    bool                enabled_;
    std::vector<STrack> tracks_;
    cv::Mat             depth_map_;
    std::string         window_name_;

    // 打印目标详细信息
    void printTargetInfo(const STrack & track) const {
        int                        class_id = track.class_id;
        int                        track_id = track.track_id;
        const std::vector<float> & tlwh     = track.tlwh;

        // 获取深度值
        int cx = static_cast<int>(tlwh[0] + tlwh[2] / 2);
        int cy = static_cast<int>(tlwh[1] + tlwh[3] / 2);
        cx     = std::max(0, std::min(cx, depth_map_.cols - 1));
        cy     = std::max(0, std::min(cy, depth_map_.rows - 1));

        float depth = 0.0f;
        if (depth_map_.type() == CV_32FC1) {
            depth = depth_map_.at<float>(cy, cx);
        } else if (depth_map_.type() == CV_8UC1) {
            depth = static_cast<float>(depth_map_.at<uchar>(cy, cx));
        }

        std::cout << "\n=== Target Info ===" << std::endl;
        std::cout << "Class: " << vClassNames[class_id] << std::endl;
        std::cout << "Track ID: " << track_id << std::endl;
        std::cout << "Depth: " << depth << std::endl;
        std::cout << "BBox: [" << tlwh[0] << ", " << tlwh[1] << ", " << tlwh[0] + tlwh[2] << ", "
                  << tlwh[1] + tlwh[3] << "]" << std::endl;
        std::cout << "==================\n" << std::endl;
    }

    // 处理鼠标点击
    void handleMouseClick(int x, int y) {
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

public:
    DisplayManager(bool enabled, const std::string & window_name = "out_frame") :
        enabled_(enabled),
        window_name_(window_name) {
        if (enabled_) {
            cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
            cv::resizeWindow(window_name_, 1280, 720);
            cv::setMouseCallback(window_name_, onMouse, this);
            std::cout << "DisplayManager initialized" << std::endl;
        }
    }

    ~DisplayManager() {
        if (enabled_) {
            cv::destroyWindow(window_name_);
        }
    }

    // 更新跟踪数据供鼠标回调使用
    void updateData(const std::vector<STrack> & tracks, const cv::Mat & depth_map) {
        if (enabled_) {
            tracks_ = tracks;
            if (!depth_map.empty()) {
                depth_map_ = depth_map.clone();
            }
        }
    }

    // 显示图像
    void show(const cv::Mat & frame) {
        if (enabled_) {
            cv::imshow(window_name_, frame);
        }
    }

    // 处理键盘事件（空格暂停）
    int handleKey(int key) {
        if (!enabled_) {
            return key;  // 返回原始按键
        }

        if (key == ' ') {
            std::cout << "Paused, press SPACE to continue..." << std::endl;
            while (true) {
                char pause_key = cv::waitKey(0);
                if (pause_key == ' ') {
                    std::cout << "Resuming..." << std::endl;
                    break;
                } else if (pause_key == 27) {  // ESC键退出
                    std::cout << "User exited" << std::endl;
                    return -404;
                }
            }
            return 0;
        } else if (key == 27) {  // ESC键退出
            std::cout << "User exited" << std::endl;
            return -404;
        }
        return key;
    }

    // 等待按键（用于刷新窗口）
    int waitKey(int delay = 1) {
        if (!enabled_) {
            return -404;
        }
        return cv::waitKey(delay);
    }

    bool isEnabled() const { return enabled_; }

    // 友元函数：鼠标回调
    friend void onMouse(int event, int x, int y, int flags, void* userdata);
};

// 全局鼠标回调函数
void onMouse(int event, int x, int y, int flags, void* userdata) {
    DisplayManager* dm = static_cast<DisplayManager*>(userdata);
    if (!dm || !dm->isEnabled()) return;

    if (event == cv::EVENT_LBUTTONDOWN) {
        dm->handleMouseClick(x, y);
    }
}