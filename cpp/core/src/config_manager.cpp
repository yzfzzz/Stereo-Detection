#include "config_manager.h"

ConfigManager::ConfigManager(std::string config_path) {
    config_     = YAML::LoadFile(config_path);
    is_display_ = config_["display_manager"]["is_display"].as<bool>(false);

    yolo_trt_file_    = config_["yolo"]["yolo_engine"].as<std::string>();
    yolo_nms_thresh_  = config_["yolo"]["yolo_nms_thresh"].as<float>(0.4f);
    yolo_conf_thresh_ = config_["yolo"]["yolo_conf_thresh"].as<float>(0.25f);

    depth_trt_file_ = config_["depth"]["depth_engine"].as<std::string>();
    depth_interval_ = config_["depth"]["depth_interval"].as<int>(1);

    is_save_   = config_["io_manager"]["is_save"].as<bool>(false);
    save_mode_ = config_["io_manager"]["save_mode"].as<std::string>("none");
    out_dir_   = config_["io_manager"]["out_dir"].as<std::string>("out_dir");

    // 运动状态引擎相关配置
    motion_velocity_threshold_ =
        config_["motion_state_engine"]["velocity_threshold"].as<float>(5.0f);
    motion_acceleration_threshold_ =
        config_["motion_state_engine"]["acceleration_threshold"].as<float>(1.5f);
    kf_process_noise_cov_ = config_["motion_state_engine"]["kf_process_noise_cov"].as<float>(2e-2f);
    kf_measurement_noise_cov_ =
        config_["motion_state_engine"]["kf_measurement_noise_cov"].as<float>(5e-2f);
}

std::string ConfigManager::getYoloEnginePath() const {
    return yolo_trt_file_;
}

std::string ConfigManager::getDepthEnginePath() const {
    return depth_trt_file_;
}

int ConfigManager::getDepthInterval() const {
    return depth_interval_;
}

std::string ConfigManager::getSaveMode() const {
    return save_mode_;
}

std::string ConfigManager::getOutDir() const {
    return out_dir_;
}

bool ConfigManager::isDisplayEnabled() const {
    return is_display_;
}

bool ConfigManager::isSaveEnabled() const {
    return is_save_;
}

float ConfigManager::getMotionVelocityThreshold() const {
    return motion_velocity_threshold_;
}

float ConfigManager::getMotionAccelerationThreshold() const {
    return motion_acceleration_threshold_;
}

float ConfigManager::getYoloNmsThresh() const {
    return yolo_nms_thresh_;
}

float ConfigManager::getYoloConfThresh() const {
    return yolo_conf_thresh_;
}

float ConfigManager::getKfProcessNoiseCov() const {
    return kf_process_noise_cov_;
}

float ConfigManager::getKfMeasurementNoiseCov() const {
    return kf_measurement_noise_cov_;
}
