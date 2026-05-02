#include "BYTETracker.h"
#include "depth_anything.h"
#include "gpu_monitor.h"
#include "infer.h"
#include "lite_mono.h"
#include "scope_timer.h"

#include <yaml-cpp/yaml.h>

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

int run(char * videoPath) {
    GpuMemoryMonitor gpu_monitor;
    gpu_monitor.PrintMemoryUsage();
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

    cv::VideoWriter writer("result.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(img_w, img_h * 2));

    YAML::Node  config         = YAML::LoadFile("config.yaml");
    std::string yolo_trt_file  = config["yolo_engine"].as<std::string>();
    std::string depth_trt_file = config["depth_engine"].as<std::string>();

    // YOLOv8 predictor
    gpu_monitor.PrintMemoryUsage();
    YoloDetector detector(yolo_trt_file, 0, 0.45, 0.01);
    gpu_monitor.PrintMemoryUsage();

    std::unique_ptr<BaseDepthModel> depth_model;
    Logger                          logger;

    gpu_monitor.PrintMemoryUsage();
    if (depth_trt_file.find("depth_anything") != std::string::npos) {
        depth_model = std::make_unique<DepthAnything>();
        cout << "Using Depth-Anything depth engine." << endl;
    } else if (depth_trt_file.find("lite_mono") != std::string::npos) {
        depth_model = std::make_unique<LiteMono>();
        cout << "Using Lite-Mono depth engine." << endl;
    } else {
        std::cerr << "Unknown depth engine type: " << depth_trt_file << std::endl;
        return -1;
    }
    depth_model->Init(depth_trt_file, logger);
    gpu_monitor.PrintMemoryUsage();

    // ByteTrack tracker
    BYTETracker tracker(fps, 30);

    cv::Mat   img;
    int       num_frames = 0;
    long long total_us   = 0;

    const int    depth_interval = 1;    // 可改为 3
    const double depth_alpha    = 0.7;  // 新深度权重，0.6~0.8 常用
    cv::Mat      cached_depth;
    bool         has_cached_depth = false;
    cv::Mat      result_depth;
    cv::Mat      depth_vis;

    cv::Mat out_frame;

    struct TrackHistory {
        std::deque<float> depths;      // 保存历史深度值
        std::deque<float> velocities;  // 保存历史速度(深度差分)
    };

    std::map<int, TrackHistory> track_history_map;

    while (true) {
        ScopedTimer timer_total("One frame average time");
        {
            ScopedTimer timer("Cap read");
            if (!cap.read(img)) {
                break;
            }
        }
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
        {
            ScopedTimer timer("Depth inference");
            bool        do_depth = (!has_cached_depth) || ((num_frames - 1) % depth_interval == 0);
            if (do_depth) {
                std::pair<cv::Mat, cv::Mat> depth_infer_result = depth_model->Predict(img);
                result_depth                                   = depth_infer_result.first;
                depth_vis                                      = depth_infer_result.second;
                has_cached_depth                               = true;
            }
        }

        // yolo inference
        std::vector<Detection> res;
        {
            ScopedTimer timer("YOLO inference");
            res = detector.inference(img);
        }

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
        std::vector<STrack> output_stracks;
        {
            ScopedTimer timer("ByteTrack");
            output_stracks = tracker.update(objects);
        }

        auto end = std::chrono::system_clock::now();
        total_us += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // Only for visualization, not required for depth estimation or tracking logic
        {
            ScopedTimer timer("Draw");
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

                // cout << "Track ID: " << track_id << " Class ID: " << class_id << " Depth: " << current_depth << endl;

                // 2. 更新该 target 的历史记录
                auto & history = track_history_map[track_id];
                history.depths.push_back(current_depth);
                if (history.depths.size() > 5) {  // 保留近 5 帧深度
                    history.depths.pop_front();
                }

                // 3. 计算速度和加速度趋势
                float       velocity_trend = 0.0f;
                float       accel_trend    = 0.0f;
                std::string motion_status  = "Stable";

                if (history.depths.size() >= 2) {
                    // 速度 = 当前深度 - 上一帧深度
                    velocity_trend = history.depths.back() - history.depths[history.depths.size() - 2];
                    history.velocities.push_back(velocity_trend);
                    if (history.velocities.size() > 5) {
                        history.velocities.pop_front();
                    }

                    if (history.velocities.size() >= 2) {
                        // 加速度 = 当前速度 - 上一帧速度
                        accel_trend = history.velocities.back() - history.velocities[history.velocities.size() - 2];
                    }

                    /* 
                       分析物理趋势：
                       注意：DepthAnything和LiteMono 等网络输出的往往是“逆深度 (disparity-like)”：
                       即：距离越近，像素值越大；距离越远，像素值越小。（请根据实际可视化确认）
                       如果你的模型是 `值大 = 距离近`，则：
                       - velocity > 0  => 值在变大 => 正在靠近
                       - velocity < 0  => 值在变小 => 正在远离
                    */
                    float noise_thresh =
                        5.0f;  // 阈值，用来过滤深度图单帧抖动产生的噪声，需要按你实际的相对深度刻度调整

                    if (velocity_trend > noise_thresh) {
                        motion_status = "Approaching";
                    } else if (velocity_trend < -noise_thresh) {
                        motion_status = "Moving away";
                    }

                    // 利用加速度判断是否急加速/急减速
                    if (std::abs(velocity_trend) > noise_thresh && std::abs(accel_trend) > (noise_thresh * 0.5f)) {
                        if ((velocity_trend > 0 && accel_trend > 0) || (velocity_trend < 0 && accel_trend < 0)) {
                            motion_status += " (Accel)";  // 正在加速靠近/远离
                        } else {
                            motion_status += " (Decel)";  // 正在减速靠近/远离
                        }
                    }
                }

                cv::Scalar  s = tracker.get_color(output_stracks[i].track_id);
                std::string label =
                    cv::format("%s #%d [%s]", vClassNames[class_id].c_str(), track_id, motion_status.c_str());

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
        }

        out_frame.create(img.rows + depth_vis.rows, img.cols, img.type());
        img.copyTo(out_frame(cv::Rect(0, 0, img.cols, img.rows)));

        if (!depth_vis.empty()) {
            if (depth_vis.size() != img.size()) {
                cv::resize(depth_vis, depth_vis, img.size());
            }
            depth_vis.copyTo(out_frame(cv::Rect(0, img.rows, img.cols, depth_vis.rows)));
        }

        // 再做拼接和写盘（只写一次）
        {
            ScopedTimer timer("Write");
            writer.write(out_frame);
        }

        // cv::imshow("img", img);
        // char c = waitKey(1);
        // if (c > 0) break;
    }

    cap.release();
    std::cout << "Infer Engine Compute FPS: " << (total_us > 0 ? (num_frames * 1000000LL / total_us) : 0) << std::endl;
    for (auto & kv : scoped_timers) {
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
