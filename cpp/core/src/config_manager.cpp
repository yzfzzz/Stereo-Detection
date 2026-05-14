#include "config_manager.h"

ConfigManager::ConfigManager(std::string config_path) {
    config                        = YAML::LoadFile(config_path);
    yolo_trt_file                 = config["yolo_engine"].as<std::string>();
    depth_trt_file                = config["depth_engine"].as<std::string>();
    depth_interval                = config["depth_interval"].as<int>(1);
    save_mode                     = config["save_mode"].as<std::string>("none");
    out_dir                       = config["out_dir"].as<std::string>("out_dir");
    is_display                    = config["is_display"].as<bool>(false);
    // 运动状态引擎相关配置
    motion_sma_window_size        = config["motion_state_engine"]["sma_window_size"].as<int>(5);
    motion_velocity_threshold     = config["motion_state_engine"]["velocity_threshold"].as<float>(5.0f);
    motion_acceleration_threshold = config["motion_state_engine"]["acceleration_threshold"].as<float>(1.5f);
}

std::string ConfigManager::GetYoloEnginePath() const {
    return yolo_trt_file;
}

std::string ConfigManager::GetDepthEnginePath() const {
    return depth_trt_file;
}

int ConfigManager::GetDepthInterval() const {
    return depth_interval;
}

std::string ConfigManager::GetSaveMode() const {
    return save_mode;
}

std::string ConfigManager::GetOutDir() const {
    return out_dir;
}

bool ConfigManager::IsDisplayEnabled() const {
    return is_display;
}

int ConfigManager::getMotionSmaWindowSize() const {
    return motion_sma_window_size;
}

float ConfigManager::getMotionVelocityThreshold() const {
    return motion_velocity_threshold;
}

float ConfigManager::getMotionAccelerationThreshold() const {
    return motion_acceleration_threshold;
}
