#pragma once
#include <yaml-cpp/yaml.h>

#include <string>

// 框架读取配置文件类
class ConfigManager {
  public:
    //  复杂解析加载配置文件
    ConfigManager(std::string config_path = "config.yaml");

    std::string getYoloEnginePath() const;

    std::string getDepthEnginePath() const;

    int getDepthInterval() const;

    std::string getSaveMode() const;

    std::string getOutDir() const;

    bool  isDisplayEnabled() const;
    bool  isSaveEnabled() const;
    
    float getMotionVelocityThreshold() const;
    float getMotionAccelerationThreshold() const;

    float getYoloNmsThresh() const;
    float getYoloConfThresh() const;
    float getKfProcessNoiseCov() const;
    float getKfMeasurementNoiseCov() const;

  private:
    YAML::Node  config;
    std::string yolo_trt_file;
    std::string depth_trt_file;
    std::string save_mode;
    std::string out_dir;
    int         depth_interval;
    bool        is_display;
    bool        is_save;
    
    float       motion_velocity_threshold;
    float       motion_acceleration_threshold;
    
    float yolo_nms_thresh;
    float yolo_conf_thresh;
    float kf_process_noise_cov;
    float kf_measurement_noise_cov;
};
