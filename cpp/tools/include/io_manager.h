#pragma once

#include "config_manager.h"

#include <sys/stat.h>

#include <opencv2/opencv.hpp>
#include <string>

class IOManager {
  public:
    // 构造函数，传入配置管理器以获取保存模式和保存路径，同时传入视频参数
    IOManager(ConfigManager & config_manager, int fps, int img_width, int img_height);

    // 析构函数负责释放系统资源（如关闭写入器）
    ~IOManager();

    // 在每一帧处理完毕后调用此函数，根据 saveMode 自动处理图片保存和/或视频写入
    void saveFrame(const cv::Mat & frame, int num_frames);

    // 辅助函数，判断文件夹是否存在
    static bool dirExists(const std::string & path);

    // 辅助函数，递归创建文件夹
    static void makeDir(const std::string & path);

  private:
    std::string     save_mode_;
    std::string     out_dir_;
    cv::VideoWriter video_writer_;
};
