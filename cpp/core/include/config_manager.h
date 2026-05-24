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

    bool isDisplayEnabled() const;
    bool isSaveEnabled() const;

    float getMotionVelocityThreshold() const;
    float getMotionAccelerationThreshold() const;

    float getYoloNmsThresh() const;
    float getYoloConfThresh() const;
    float getKfProcessNoiseCov() const;
    float getKfMeasurementNoiseCov() const;

  private:
    YAML::Node  config_;
    std::string yolo_trt_file_;
    std::string depth_trt_file_;
    std::string save_mode_;
    std::string out_dir_;
    int         depth_interval_;
    bool        is_display_;
    bool        is_save_;

    float motion_velocity_threshold_;
    float motion_acceleration_threshold_;

    float yolo_nms_thresh_;
    float yolo_conf_thresh_;
    float kf_process_noise_cov_;
    float kf_measurement_noise_cov_;
};
