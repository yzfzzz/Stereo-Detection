#pragma once

#include "STrack.h"

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// 前向声明
class DisplayManager;

// 全局变量声明（来自 config.h）
extern const std::vector<std::string> vClassNames;

enum Key_Input {
    ESC   = 27,
    SPACE = 32,
    LEFT  = 65361,
    UP    = 65362,
    RIGHT = 65363,
    DOWN  = 65364,
    A     = 97,
    D     = 100,
    S     = 115,
    W     = 119,
    Q     = 113,
    E     = 101,
    R     = 114,
    Z     = 122,
};

// 显示管理器类：负责窗口管理、显示、鼠标点击等事件
class DisplayManager {
  private:
    bool                enabled_;
    std::vector<STrack> tracks_;
    cv::Mat             depth_map_;
    std::string         window_name_;

    // 在 bounding box 内采样多个点并计算深度均值（鲁棒性更好）
    float computeMeanDepth(const std::vector<float> & tlwh) const;

    // 打印目标详细信息
    void printTargetInfo(const STrack & track) const;

    // 处理鼠标点击
    void handleMouseClick(int x, int y);

  public:
    DisplayManager(bool enabled, const std::string & window_name = "out_frame");
    ~DisplayManager();

    // 更新跟踪数据供鼠标回调使用
    void updateData(const std::vector<STrack> & tracks, const cv::Mat & depth_map);

    // 显示图像
    void show(const cv::Mat & frame);

    // 处理键盘事件（空格暂停）
    int handleKey(int key);
    // 等待按键（用于刷新窗口）
    int waitKey(int delay = 1);

    bool isEnabled() const { return enabled_; }

    // 友元函数：鼠标回调
    friend void onMouse(int event, int x, int y, int flags, void * userdata);
};
