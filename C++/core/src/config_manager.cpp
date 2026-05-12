#include "config_manager.h"

ConfigManager::ConfigManager(std::string config_path) {
    config         = YAML::LoadFile(config_path);
    yolo_trt_file  = config["yolo_engine"].as<std::string>();
    depth_trt_file = config["depth_engine"].as<std::string>();
    depth_interval = config["depth_interval"].as<int>(1);
    save_mode      = config["save_mode"].as<std::string>("none");
    out_dir        = config["out_dir"].as<std::string>("out_dir");
    is_despaly     = config["is_despaly"].as<bool>(false);
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
    return is_despaly;
}
