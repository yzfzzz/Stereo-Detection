#include "BYTETracker.h"
#include "config.h"
#include "config_manager.h"
#include "depth_anything.h"
#include "infer.h"
#include "io_manager.h"
#include "lite_mono.h"
#include "motion_state_engine.h"
#include "pipeline.h"
#include "scope_timer.h"
#include "visual_manager.h"

#include <dlfcn.h>
#include <sys/stat.h>

#include <array>
#include <iostream>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <string>
#include <utility>
#include <vector>

cv::Mat draw_one_frame(cv::Mat &                  img,
                       const FrameResult &        frame_result,
                       const ConfigManager &      config_manager,
                       DrawingManager &           drawing_manager,
                       std::function<Scalar(int)> get_color_func,
                       int                        num_frames,
                       int                        total_us) {
    for (int i = 0; i < frame_result.tracked_objects.size(); i++) {
        auto & track = frame_result.tracked_objects[i];
        if (track.tlwh[2] * track.tlwh[3] <= 20) {
            continue;
        }

        auto it = frame_result.motion_records.find(track.track_id);
        if (it != frame_result.motion_records.end()) {
            drawing_manager.drawTrackedObject(img, track, it->second, get_color_func(track.track_id));
        }
    }
    // FPS
    int show_fps = (total_us > 0) ? (num_frames * 1000000LL / total_us) : 0;
    // 全局信息
    drawing_manager.drawGlobalInfo(img, num_frames, show_fps, frame_result.tracked_objects.size());

    // 上下拼接
    cv::Mat out_frame = drawing_manager.concatenateFrames(img, frame_result.depth_vis);
    return out_frame;
}

int run(char * videoPath) {
    // read video
    std::string      input_video_path = std::string(videoPath);
    cv::VideoCapture cap(input_video_path);
    if (!cap.isOpened()) {
        return 0;
    }
    int  img_w            = cap.get(CAP_PROP_FRAME_WIDTH);
    int  img_h            = cap.get(CAP_PROP_FRAME_HEIGHT);
    int  fps              = cap.get(CAP_PROP_FPS);
    long total_frames_num = static_cast<long>(cap.get(CAP_PROP_FRAME_COUNT));
    cout << "Total frames: " << total_frames_num << endl;

    // 读取配置文件
    ConfigManager  config_manager("config.yaml");
    // 推理流水线（负责目标检测、深度估计、跟踪、运动状态判断等核心功能）
    Pipeline       pipeline(config_manager);
    // 文件读写，落盘保存
    IOManager      io_manager(config_manager, fps, img_w, img_h * 2);
    // 绘制管理器（负责绘制结果）
    DrawingManager drawing_manager(vClassNames);
    // 显示管理器（负责窗口管理、显示、鼠标点击等）
    DisplayManager display_manager(config_manager.isDisplayEnabled(), "Detection Result", cv::Size(img_w, img_h * 2));

    int       num_frames = 0;
    long long total_us   = 0;
    cv::Mat   img;

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
        if (img.empty()) {
            break;
        }
        num_frames++;
        if (num_frames % 100 == 0) {
            cout << "Processing frame " << num_frames << " ("
                 << (total_us > 0 ? (num_frames * 1000000LL / total_us) : 0) << " fps)" << endl;
        }

        auto start = std::chrono::system_clock::now();

        // 执行推理流水线
        FrameResult frame_result;
        pipeline.process(img, num_frames, frame_result);

        auto end = std::chrono::system_clock::now();
        total_us += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // 画图
        cv::Mat out_frame = draw_one_frame(
            img, frame_result, config_manager, drawing_manager,
            [&pipeline](int idx) { return pipeline.get_color(idx); }, num_frames, total_us);
        io_manager.saveFrame(out_frame, num_frames);

        // 显示图像（通过 DisplayManager）
        display_manager.updateData(frame_result.tracked_objects, frame_result.result_depth);
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
