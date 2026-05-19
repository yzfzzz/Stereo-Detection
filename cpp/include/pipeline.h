#pragma once
#include "BYTETracker.h"
#include "config_manager.h"
#include "depth_anything.h"
#include "infer.h"
#include "io_manager.h"
#include "lite_mono.h"
#include "motion_state_engine.h"
#include "visual_manager.h"

struct FrameResult {
    std::vector<STrack>                            tracked_objects;
    cv::Mat                                        result_depth;
    cv::Mat                                        depth_vis;
    std::unordered_map<int, MotionStateInfoRecord> motion_records;
};

class Pipeline {
  public:
    Pipeline(ConfigManager config_manager);

    // 核心推理接口，供正常业务和 Benchmark 调用
    void process(cv::Mat & img, int num_frames, FrameResult & out_result);

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
