#include "BYTETracker.h"
#include "config_manager.h"
#include "depth_anything.h"
#include "infer.h"
#include "io_manager.h"
#include "lite_mono.h"
#include "motion_state_engine.h"
#include "scope_timer.h"
#include "visual_manager.h"

#include <dlfcn.h>
#include <sys/stat.h>

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// 需要跟踪的类别，可以根据自己需求调整，筛选自己想要跟踪的对象的种类（以下对应COCO数据集类别索引）
std::vector<int> gTrackClasses{ 1, 2, 3, 5, 7 };  // person, bicycle, car, motorcycle, bus, truck

int run(char * videoPath) {
    // read video
    std::string      input_video_path = std::string(videoPath);
    cv::VideoCapture cap(input_video_path);
    if (!cap.isOpened()) {
        return 0;
    }
    int  img_w   = cap.get(CAP_PROP_FRAME_WIDTH);
    int  img_h   = cap.get(CAP_PROP_FRAME_HEIGHT);
    int  fps     = cap.get(CAP_PROP_FPS);
    long n_frame = static_cast<long>(cap.get(CAP_PROP_FRAME_COUNT));
    cout << "Total frames: " << n_frame << endl;

    // =========ConfigManager 读取配置文件=========
    ConfigManager  config_manager("config.yaml");
    // 文件读写，落盘保存
    IOManager      io_manager(config_manager, fps, img_w, img_h * 2);
    // 绘制管理器（负责绘制结果）
    DrawingManager drawing_manager(vClassNames);
    // 显示管理器（负责窗口管理、显示、鼠标点击等）
    DisplayManager display_manager(config_manager.isDisplayEnabled(), "Detection Result", cv::Size(img_w, img_h * 2));
    // 运动状态引擎（负责计算运动状态）
    MotionStateEngine motion_state_engine(config_manager.getMotionVelocityThreshold(),
                                          config_manager.getMotionAccelerationThreshold());

    // =========YOLOv8 predictor=========
    Logger                          logger;
    YoloDetector                    detector(config_manager.getYoloEnginePath(), 0, 0.35, 0.1);
    // =========Depth predictor=========
    std::unique_ptr<BaseDepthModel> depth_model;
    if (config_manager.getDepthEnginePath().find("depth_anything") != std::string::npos) {
        depth_model = std::make_unique<DepthAnything>();
        cout << "Using Depth-Anything depth engine." << endl;
    } else if (config_manager.getDepthEnginePath().find("lite_mono") != std::string::npos) {
        depth_model = std::make_unique<LiteMono>();
        cout << "Using Lite-Mono depth engine." << endl;
    } else {
        std::cerr << "Unknown depth engine type: " << config_manager.getDepthEnginePath() << std::endl;
        return -1;
    }
    depth_model->Init(config_manager.getDepthEnginePath(), logger);

    // ByteTrack tracker
    BYTETracker tracker(fps, 30);

    int       num_frames       = 0;
    long long total_us         = 0;
    bool      has_cached_depth = false;

    cv::Mat img, cached_depth, result_depth, depth_vis, out_frame;

    auto is_tracking_class = [](int class_id) {
        for (auto & c : gTrackClasses) {
            if (class_id == c) {
                return true;
            }
        }
        return false;
    };

    while (true) {
#if defined(ENABLE_TIMER)
        if (!DEBUG_FUNCTION_RUNNING_TIME_MEMBER_REF("1.Cap Read", cap, read, img)) {
            break;
        }
#else
        if (!cap.read(img)) {
            break;
        }
#endif

        num_frames++;
        if (num_frames % 100 == 0) {
            cout << "Processing frame " << num_frames << " ("
                 << (total_us > 0 ? (num_frames * 1000000LL / total_us) : 0) << " fps)" << endl;
        }
        if (img.empty()) {
            break;
        }

        auto start = std::chrono::system_clock::now();

        // 端侧部署的情况下可以用串行保证端到端低延迟
        // depthinference
        bool do_depth = (!has_cached_depth) || ((num_frames - 1) % config_manager.getDepthInterval() == 0);
        if (do_depth) {
#if defined(ENABLE_TIMER)
            std::pair<cv::Mat, cv::Mat> depth_infer_result =
                DEBUG_FUNCTION_RUNNING_TIME_MEMBER_REF("3.Depth Infer", *depth_model, Predict, img);
#else
            std::pair<cv::Mat, cv::Mat> depth_infer_result = depth_model->Predict(img);
#endif
            result_depth     = depth_infer_result.first;
            depth_vis        = depth_infer_result.second;
            has_cached_depth = true;
        }

        // yolo inference
#if defined(ENABLE_TIMER)
        std::vector<Detection> res = DEBUG_FUNCTION_RUNNING_TIME_MEMBER_REF("2.YOLO Infer", detector, inference, img);
#else
        std::vector<Detection> res = detector.inference(img);

#endif
        // yolo output format to bytetrack input format, and filter bbox by class id
        std::vector<Object> objects;
        for (size_t j = 0; j < res.size(); j++) {
            float * bbox    = res[j].bbox;
            float   conf    = res[j].conf;
            int     classId = res[j].classId;

            if (is_tracking_class(classId)) {
                cv::Rect_<float> rect(bbox[0], bbox[1], (bbox[2] - bbox[0]), (bbox[3] - bbox[1]));
                Object           obj{ rect, classId, conf };
                objects.push_back(obj);
            }
        }

        // track
#if defined(ENABLE_TIMER)
        std::vector<STrack> output_stracks =
            DEBUG_FUNCTION_RUNNING_TIME_MEMBER_REF("4.ByteTrack", tracker, update, objects);
#else
        std::vector<STrack> output_stracks = tracker.update(objects);

#endif
        // 更新显示管理器数据（供鼠标点击查询使用）
        display_manager.updateData(output_stracks, result_depth);

        auto end = std::chrono::system_clock::now();
        total_us += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // Only for visualization, not required for depth estimation or tracking logic
        for (int i = 0; i < output_stracks.size(); i++) {
            if (output_stracks[i].tlwh[2] * output_stracks[i].tlwh[3] <= 20) {
                continue;
            }

            int class_id = output_stracks[i].class_id;
            int track_id = output_stracks[i].track_id;

            float  current_depth   = motion_state_engine.getObjectDepth(result_depth, output_stracks[i], img.size());
            // 获取时间戳 (秒)
            double frame_timestamp = std::chrono::duration<double>(end.time_since_epoch()).count();

            // 计算运动状态
            MotionStateInfoRecord motion_state_info_record =
                motion_state_engine.computeMotionState(track_id, current_depth, frame_timestamp);

            drawing_manager.drawTrackedObject(img, output_stracks[i], motion_state_info_record,
                                              tracker.get_color(output_stracks[i].track_id));
        }

        // FPS
        int show_fps = (total_us > 0) ? (num_frames * 1000000LL / total_us) : 0;

        // 全局信息
        drawing_manager.drawGlobalInfo(img, num_frames, show_fps, output_stracks.size());

        // 上下拼接
        cv::Mat out_frame = drawing_manager.concatenateFrames(img, depth_vis);

        io_manager.saveFrame(out_frame, num_frames);

        // 显示图像（通过 DisplayManager）
        if (display_manager.isEnabled()) {
            display_manager.show(out_frame);
            char c      = display_manager.waitKey(1);
            int  result = display_manager.handleKey(c);
            if (result == Key_Input::ESC) {  // 用户按下 ESC 键退出
                break;                       // 用户退出
            }
        }
    }
    cap.release();

    std::cout << "==========Summary===========" << endl;
    std::cout << "Infer Engine Compute FPS: " << (total_us > 0 ? (num_frames * 1000000LL / total_us) : 0) << std::endl;
    for (auto & kv : ScopedTimer::GetScopedTimers()) {
        std::cout << kv.first << ": " << (kv.second / 1000.0) / num_frames << " ms/frame" << std::endl;
    }

    return 0;
}

int main(int argc, char * argv[]) {
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "Usage: ./main [video path]" << std::endl;
        std::cerr << "Example: ./main ./videos/demo.mp4" << std::endl;
        return -1;
    }

    return run(argv[1]);
}
