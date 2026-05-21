#include "pipeline.h"

#include "BYTETracker.h"
#include "config_manager.h"
#include "depth_anything.h"
#include "frame.h"
#include "lite_mono.h"
#include "motion_state_engine.h"
#include "scope_timer.h"

#include <memory>

Pipeline::Pipeline(ConfigManager config_manager, FrameMeta frame_meta) :
    config_manager_(config_manager),
    detector_(config_manager_.getYoloEnginePath(),
              frame_meta.img_w,
              frame_meta.img_h,
              config_manager_.getYoloNmsThresh(),
              config_manager_.getYoloConfThresh()),
    tracker_(30, 30),  // 假设fps=30，或从config读取
    motion_state_engine_(config_manager_.getMotionVelocityThreshold(),
                         config_manager_.getMotionAccelerationThreshold(),
                         config_manager_.getKfProcessNoiseCov(),
                         config_manager_.getKfMeasurementNoiseCov()) {
    if (config_manager_.getDepthEnginePath().find("depth_anything") != std::string::npos) {
        depth_model_ = std::make_unique<DepthAnything>();
    } else if (config_manager_.getDepthEnginePath().find("lite") != std::string::npos) {
        depth_model_ = std::make_unique<LiteMono>();
    }
    depth_model_->Init(config_manager_.getDepthEnginePath(), frame_meta.img_w, frame_meta.img_h);
}

// 同步
void Pipeline::process(FrameInputContext & frame_input_context, InferOutputContext & infer_output_context) {
    bool do_depth =
        (!has_cached_depth_) || ((frame_input_context.frame_id - 1) % config_manager_.getDepthInterval() == 0);

    if (do_depth) {
        auto depth_infer_result = depth_model_->Predict(frame_input_context.raw_img);

        infer_output_context.result_depth = depth_infer_result.first;
        infer_output_context.depth_vis    = depth_infer_result.second;
        has_cached_depth_                 = true;
    } else {
        infer_output_context.result_depth = cached_depth_;
        infer_output_context.depth_vis    = cached_depth_vis_;
    }

    std::vector<Detection> res = detector_.inference(frame_input_context.raw_img);
    postProcess(frame_input_context, infer_output_context, res);
}

void Pipeline::processAsync(FrameInputContext & frame_input_context, InferOutputContext & infer_output_context) {
    bool do_depth =
        (!has_cached_depth_) || ((frame_input_context.frame_id - 1) % config_manager_.getDepthInterval() == 0);

    if (do_depth) {
        depth_model_->PredictAsync(frame_input_context.raw_img);
    }
    detector_.inferenceAsync(frame_input_context.raw_img);

    depth_model_->WaitAsync();
    detector_.WaitAsync();
    if (do_depth) {
        infer_output_context.result_depth = depth_model_->GetPredictResultAsync().first;
        infer_output_context.depth_vis    = depth_model_->GetPredictResultAsync().second;
    } else {
        infer_output_context.result_depth = cached_depth_;
        infer_output_context.depth_vis    = cached_depth_vis_;
    }
    std::vector<Detection> res = detector_.GetInferResultAsync(frame_input_context.raw_img);

    postProcess(frame_input_context, infer_output_context, res);
}

void Pipeline::postProcess(FrameInputContext &            frame_input_context,
                           InferOutputContext &           infer_output_context,
                           const std::vector<Detection> & res) {
    std::vector<Object> objects;
    for (size_t j = 0; j < res.size(); j++) {
        if (isTrackingClass(res[j].classId)) {
            cv::Rect_<float> rect(res[j].bbox[0], res[j].bbox[1], (res[j].bbox[2] - res[j].bbox[0]),
                                  (res[j].bbox[3] - res[j].bbox[1]));
            objects.push_back({ rect, res[j].classId, res[j].conf });
        }
    }

    infer_output_context.tracked_objects = tracker_.update(objects);

    for (int i = 0; i < infer_output_context.tracked_objects.size(); i++) {
        if (infer_output_context.tracked_objects[i].tlwh[2] * infer_output_context.tracked_objects[i].tlwh[3] <= 20) {
            continue;
        }
        int track_id = infer_output_context.tracked_objects[i].track_id;

        float current_depth = motion_state_engine_.getObjectDepth(infer_output_context.result_depth,
                                                                  infer_output_context.tracked_objects[i],
                                                                  frame_input_context.raw_img.size());

        infer_output_context.motion_records.insert(
            { track_id,
              motion_state_engine_.computeMotionState(track_id, current_depth, frame_input_context.timestamp) });
    }

    // 更新缓存
    cached_depth_     = infer_output_context.result_depth;
    cached_depth_vis_ = infer_output_context.depth_vis;
}
