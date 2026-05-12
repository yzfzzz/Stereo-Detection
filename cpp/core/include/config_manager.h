#pragma once
#include <yaml-cpp/yaml.h>

#include <string>

// 框架读取配置文件类
class ConfigManager {
  public:
    //  复杂解析加载配置文件
    ConfigManager(std::string config_path = "config.yaml");

    std::string GetYoloEnginePath() const;

    std::string GetDepthEnginePath() const;

    int GetDepthInterval() const;

    std::string GetSaveMode() const;

    std::string GetOutDir() const;

    bool IsDisplayEnabled() const;


  private:
    YAML::Node  config;
    std::string yolo_trt_file;
    std::string depth_trt_file;
    std::string save_mode;
    std::string out_dir;
    int         depth_interval;
    bool        is_display;
};
