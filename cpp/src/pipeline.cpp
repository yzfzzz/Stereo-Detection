#include "pipeline.h"

#include "BYTETracker.h"
#include "config_manager.h"
#include "motion_state_engine.h"
#include "scope_timer.h"

#include <memory>

Pipeline::Pipeline(ConfigManager config_manager) :
    config_manager_(config_manager),
    detector_(config_manager_.getYoloEnginePath(),
              0,
              config_manager_.getYoloNmsThresh(),
              config_manager_.getYoloConfThresh()),
    tracker_(30, 30),  // 假设fps=30，或从config读取
    motion_state_engine_(config_manager_.getMotionVelocityThreshold(),
                         config_manager_.getMotionAccelerationThreshold(),
                         config_manager_.getKfProcessNoiseCov(),
                         config_manager_.getKfMeasurementNoiseCov()) {
    initDepthModel();
}

void Pipeline::initDepthModel() {
    if (config_manager_.getDepthEnginePath().find("depth_anything") != std::string::npos) {
        depth_model_ = std::make_unique<DepthAnything>();
    } else if (config_manager_.getDepthEnginePath().find("lite") != std::string::npos) {
        depth_model_ = std::make_unique<LiteMono>();
    }
    depth_model_->Init(config_manager_.getDepthEnginePath());
}

void Pipeline::process(cv::Mat & img, int num_frames, FrameResult & out_result) {
    bool do_depth = (!has_cached_depth_) || ((num_frames - 1) % config_manager_.getDepthInterval() == 0);
    if (do_depth) {
#if defined(ENABLE_TIMER)
        auto depth_infer_result = DEBUG_FUNCTION_RUNNING_TIME_MEMBER_PTR("3.Depth", depth_model_, Predict, img);
#else
        auto depth_infer_result = depth_model_->Predict(img);
#endif
        out_result.result_depth = depth_infer_result.first;
        out_result.depth_vis    = depth_infer_result.second;
        has_cached_depth_       = true;
    } else {
        out_result.result_depth = cached_depth_;
        out_result.depth_vis    = cached_depth_vis_;
    }

#if defined(ENABLE_TIMER)
    std::vector<Detection> res = DEBUG_FUNCTION_RUNNING_TIME_MEMBER_REF("2.YOLO", detector_, inference, img);
#else
    std::vector<Detection> res = detector_.inference(img);
#endif

    std::vector<Object> objects;
    for (size_t j = 0; j < res.size(); j++) {
        if (isTrackingClass(res[j].classId)) {
            cv::Rect_<float> rect(res[j].bbox[0], res[j].bbox[1], (res[j].bbox[2] - res[j].bbox[0]),
                                  (res[j].bbox[3] - res[j].bbox[1]));
            objects.push_back({ rect, res[j].classId, res[j].conf });
        }
    }

#if defined(ENABLE_TIMER)
    out_result.tracked_objects = DEBUG_FUNCTION_RUNNING_TIME_MEMBER_REF("4.ByteTrack", tracker_, update, objects);
#else
    out_result.tracked_objects = tracker_.update(objects);
#endif

    for (int i = 0; i < out_result.tracked_objects.size(); i++) {
        if (out_result.tracked_objects[i].tlwh[2] * out_result.tracked_objects[i].tlwh[3] <= 20) {
            continue;
        }
        int    track_id        = out_result.tracked_objects[i].track_id;
        auto   end_time        = std::chrono::system_clock::now();
        double frame_timestamp = std::chrono::duration<double>(end_time.time_since_epoch()).count();

#if defined(ENABLE_TIMER)
        float current_depth = DEBUG_FUNCTION_RUNNING_TIME_MEMBER_REF(
            "5.Motion Engine getObjectDepth", motion_state_engine_, getObjectDepth, out_result.result_depth,
            out_result.tracked_objects[i], img.size());

#else
        float current_depth =
            motion_state_engine_.getObjectDepth(out_result.result_depth, out_result.tracked_objects[i], img.size());
#endif

        out_result.motion_records.insert(
            { track_id, motion_state_engine_.computeMotionState(track_id, current_depth, frame_timestamp) });
    }

    // 更新缓存
    cached_depth_     = out_result.result_depth;
    cached_depth_vis_ = out_result.depth_vis;
}
