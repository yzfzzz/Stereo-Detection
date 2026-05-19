
#include "config.h"
#include "config_manager.h"
#include "io_manager.h"
#include "pipeline.h"
#include "scope_timer.h"
#include "visual_manager.h"

#include <dlfcn.h>
#include <sys/stat.h>

#include <cstdio>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
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
#if defined(ENABLE_TIMER)
            DEBUG_FUNCTION_RUNNING_TIME_MEMBER_REF("6.Drawing Manager", drawing_manager, drawTrackedObject, img, track,
                                                   it->second, get_color_func(track.track_id));
#else
            drawing_manager.drawTrackedObject(img, track, it->second, get_color_func(track.track_id));
#endif
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

int run(char * video_path) {
    // read video
    cv::VideoCapture cap(video_path);
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
    DisplayManager display_manager(config_manager, "Detection Result", cv::Size(img_w, img_h * 2));

    int     num_frames = 0;
    double  total_us   = 0;
    cv::Mat img;

    while (true) {
        num_frames++;
        if (num_frames % 100 == 0) {
            printf("Processing frame %d (%.2f fps)\n", num_frames,
                   (total_us > 0 ? (num_frames * 1000000LL / total_us) : 0));
        }
        FrameResult frame_result;
#if defined(ENABLE_TIMER)
        if (!DEBUG_FUNCTION_RUNNING_TIME_MEMBER_REF("1.Cap Read", cap, read, img) || img.empty()) {
            break;
        }
        // 执行推理流水线
        std::string name = "Infer Pipeline";
        DEBUG_FUNCTION_RUNNING_TIME_MEMBER_REF(name, pipeline, process, img, num_frames, frame_result);
        total_us += ScopedTimer::GetScopedTimers()[name].back();  // 获取刚刚这次推理的耗时
#else
        if (!cap.read(img) || img.empty()) {
            break;
        }
        pipeline.process(img, num_frames, frame_result);

#endif

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
#if defined(ENABLE_TIMER)
    std::cout << "==========Summary===========" << endl;
    for (auto & kv : ScopedTimer::GetScopedTimers()) {
        double avg = calculateAverage(kv.second);
        double p95 = calculatePercentile(kv.second, 95.0);
        double p99 = calculatePercentile(kv.second, 99.0);
        printf("[%s]: avg = %.2f ms, P95 = %.2f ms, P99 = %.2f ms (frame)\n", kv.first.c_str(), avg, p95, p99);
    }
#endif

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
