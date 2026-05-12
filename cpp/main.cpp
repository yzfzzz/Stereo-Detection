#include "BYTETracker.h"
#include "config_manager.h"
#include "depth_anything.h"
#include "display_manager.h"
#include "infer.h"
#include "lite_mono.h"
#include "scope_timer.h"

#include <dlfcn.h>
#include <sys/stat.h>

#include <iostream>
#include <memory>
#include <string>

// 需要跟踪的类别，可以根据自己需求调整，筛选自己想要跟踪的对象的种类（以下对应COCO数据集类别索引）
std::vector<int> trackClasses{ 0, 1, 2, 3, 5, 7 };  // person, bicycle, car, motorcycle, bus, truck

bool isTrackingClass(int class_id) {
    for (auto & c : trackClasses) {
        if (class_id == c) {
            return true;
        }
    }
    return false;
}

bool dirExists(const std::string & path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
}

int run(char * videoPath) {
    // read video
    std::string      inputVideoPath = std::string(videoPath);
    cv::VideoCapture cap(inputVideoPath);
    if (!cap.isOpened()) {
        return 0;
    }

    int  img_w  = cap.get(CAP_PROP_FRAME_WIDTH);
    int  img_h  = cap.get(CAP_PROP_FRAME_HEIGHT);
    int  fps    = cap.get(CAP_PROP_FPS);
    long nFrame = static_cast<long>(cap.get(CAP_PROP_FRAME_COUNT));
    cout << "Total frames: " << nFrame << endl;

    // ConfigManager 读取配置文件
    ConfigManager config_manager("config.yaml");
    if (config_manager.GetSaveMode() == "images" || config_manager.GetSaveMode() == "both") {
        if (!dirExists(config_manager.GetOutDir())) {
            system(("mkdir -p " + config_manager.GetOutDir()).c_str());
        }
    }
    cv::VideoWriter writer;
    if (config_manager.GetSaveMode() == "video" || config_manager.GetSaveMode() == "both") {
        writer.open("result.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(img_w, img_h * 2));
    }
    // YOLOv8 predictor
    YoloDetector                    detector(config_manager.GetYoloEnginePath(), 0, 0.45, 0.01);
    std::unique_ptr<BaseDepthModel> depth_model;
    Logger                          logger;

    if (config_manager.GetDepthEnginePath().find("depth_anything") != std::string::npos) {
        depth_model = std::make_unique<DepthAnything>();
        cout << "Using Depth-Anything depth engine." << endl;
    } else if (config_manager.GetDepthEnginePath().find("lite_mono") != std::string::npos) {
        depth_model = std::make_unique<LiteMono>();
        cout << "Using Lite-Mono depth engine." << endl;
    } else {
        std::cerr << "Unknown depth engine type: " << config_manager.GetDepthEnginePath() << std::endl;
        return -1;
    }
    depth_model->Init(config_manager.GetDepthEnginePath(), logger);

    // ByteTrack tracker
    BYTETracker tracker(fps, 30);

    // 创建显示管理器（负责窗口管理、显示、鼠标点击等）
    DisplayManager display_manager(config_manager.IsDisplayEnabled());

    cv::Mat   img;
    int       num_frames = 0;
    long long total_us   = 0;

    const double depth_alpha = 0.7;  // 新深度权重，0.6~0.8 常用
    cv::Mat      cached_depth;
    bool         has_cached_depth = false;
    cv::Mat      result_depth;
    cv::Mat      depth_vis;

    cv::Mat out_frame;

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
        bool do_depth = (!has_cached_depth) || ((num_frames - 1) % config_manager.GetDepthInterval() == 0);
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

            if (isTrackingClass(classId)) {
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
            const std::vector<float> & tlwh = output_stracks[i].tlwh;
            if (tlwh[2] * tlwh[3] <= 20) {
                continue;
            }

            int class_id = output_stracks[i].class_id;

            int track_id = output_stracks[i].track_id;

            // 1. 提取目标中心点在深度图上的相对深度
            int cx = static_cast<int>(tlwh[0] + tlwh[2] / 2);
            int cy = static_cast<int>(tlwh[1] + tlwh[3] / 2);

            // 防止越界
            cx = std::max(0, std::min(cx, result_depth.cols - 1));
            cy = std::max(0, std::min(cy, result_depth.rows - 1));

            float current_depth = 0.0f;
            // 根据深度图的数据类型获取数值，通常推理输出是 CV_32FC1 或归一化后的 CV_8UC1
            if (result_depth.type() == CV_32FC1) {
                current_depth = result_depth.at<float>(cy, cx);
            } else if (result_depth.type() == CV_8UC1) {
                current_depth = static_cast<float>(result_depth.at<uchar>(cy, cx));
            }

            cv::Scalar  s = tracker.get_color(output_stracks[i].track_id);
            std::string label =
                cv::format("%s #%d [Depth: %.2f]", vClassNames[class_id].c_str(), track_id, current_depth);

            int      baseLine   = 0;
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseLine);
            cv::Rect rect_bg(cv::Point((int) tlwh[0], (int) tlwh[1] - label_size.height - 8),
                             cv::Size(label_size.width + 8, label_size.height + 8));
            cv::rectangle(img, rect_bg, s, cv::FILLED);
            cv::putText(img, label, cv::Point((int) tlwh[0] + 4, (int) tlwh[1] - 4), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

            cv::rectangle(img, cv::Rect((int) tlwh[0], (int) tlwh[1], (int) tlwh[2], (int) tlwh[3]), s, 2);
        }

        int show_fps = (total_us > 0) ? (num_frames * 1000000LL / total_us) : 0;
        cv::putText(img, cv::format("frame: %d fps: %d num: %ld", num_frames, show_fps, output_stracks.size()),
                    cv::Point(0, 30), 0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

        out_frame.create(img.rows + depth_vis.rows, img.cols, img.type());
        img.copyTo(out_frame(cv::Rect(0, 0, img.cols, img.rows)));

        if (!depth_vis.empty()) {
            if (depth_vis.size() != img.size()) {
                cv::resize(depth_vis, depth_vis, img.size());
            }
            depth_vis.copyTo(out_frame(cv::Rect(0, img.rows, img.cols, depth_vis.rows)));
        }

        // 保存图像或视频
        if (config_manager.GetSaveMode() == "images" || config_manager.GetSaveMode() == "both") {
            std::string save_path = config_manager.GetOutDir() + "/frame_" + std::to_string(num_frames) + ".jpg";
            cv::imwrite(save_path, out_frame);
        }

        if (config_manager.GetSaveMode() == "video" || config_manager.GetSaveMode() == "both") {
            writer.write(out_frame);
        }

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
    if (writer.isOpened()) {
        writer.release();
    }

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
