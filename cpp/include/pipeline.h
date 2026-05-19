#pragma once
#include "base.h"
#include "BYTETracker.h"
#include "config_manager.h"
#include "frame.h"
#include "infer.h"
#include "motion_state_engine.h"

class Pipeline {
  public:
    Pipeline(ConfigManager config_manager, FrameMeta frame_meta);

    // 核心推理接口，供正常业务和 Benchmark 调用
    void process(FrameInputContext & frame_input_context, InferOutputContext & infer_output_context);

    Scalar get_color(int idx) { return tracker_.get_color(idx); }


  private:
    void initDepthModel();

    bool isTrackingClass(int class_id) {
        for (auto & c : trackClasses) {
            if (class_id == c) {
                return true;
            }
        }
        return false;
    }

    ConfigManager                   config_manager_;
    YoloDetector                    detector_;
    std::unique_ptr<BaseDepthModel> depth_model_;
    BYTETracker                     tracker_;
    MotionStateEngine               motion_state_engine_;

    // 跨帧缓存状态
    bool             has_cached_depth_ = false;
    cv::Mat          cached_depth_;
    cv::Mat          cached_depth_vis_;
    // 需要跟踪的类别，可以根据自己需求调整，筛选自己想要跟踪的对象的种类（以下对应COCO数据集类别索引）
    std::vector<int> trackClasses{ 1, 2, 3, 5, 7 };  // person, bicycle, car, motorcycle, bus, truck
};
