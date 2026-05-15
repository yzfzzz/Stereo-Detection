#include "io_manager.h"

#include <cstdlib>  // For system()
#include <iostream>

IOManager::IOManager(ConfigManager & config_manager, int fps, int img_width, int img_height) :
    save_mode_(config_manager.getSaveMode()),
    out_dir_(config_manager.getOutDir()) {
    // 如果需要保存图片，检查目标文件夹并创建
    if (save_mode_ == "images" || save_mode_ == "both") {
        if (!dirExists(out_dir_)) {
            makeDir(out_dir_);
        }
    }

    // 如果需要保存视频，初始化 VideoWriter
    if (save_mode_ == "video" || save_mode_ == "both") {
        std::string video_save_path = out_dir_ + "/result.mp4";  // 最好也放进输出目录
        video_writer_.open(video_save_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                           cv::Size(img_width, img_height));

        if (!video_writer_.isOpened()) {
            std::cerr << "[Error] Failed to initialize VideoWriter at " << video_save_path << std::endl;
        }
    }
}

IOManager::~IOManager() {
    if (video_writer_.isOpened()) {
        video_writer_.release();
    }
}

void IOManager::saveFrame(const cv::Mat & frame, int num_frames) {
    if (save_mode_ == "images" || save_mode_ == "both") {
        std::string save_path = out_dir_ + "/frame_" + std::to_string(num_frames) + ".jpg";
        cv::imwrite(save_path, frame);
    }

    if (save_mode_ == "video" || save_mode_ == "both") {
        if (video_writer_.isOpened()) {
            video_writer_.write(frame);
        }
    }
}

bool IOManager::dirExists(const std::string & path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
}

void IOManager::makeDir(const std::string & path) {
    std::string cmd = "mkdir -p " + path;
    int         ret = system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "[Warning] Could not execute mkdir completely." << std::endl;
    }
}
