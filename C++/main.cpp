#include "BYTETracker.h"
#include "depth_anything.h"
#include "infer.h"
#include "scope_timer.h"

#include <yaml-cpp/yaml.h>

#include <iostream>
#include <string>
#include <thread>

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
    YoloDetector detector(yolo_trt_file, 0, 0.45, 0.01);

    // Depth Anything predictor
    DepthAnything depth_model;
    Logger        logger;
    depth_model.init(depth_trt_file, logger);

    // ByteTrack tracker
    BYTETracker tracker(fps, 30);

    cv::Mat   img;
    int       num_frames = 0;
    long long total_us   = 0;

    const int    depth_interval = 2;    // 可改为 3
    const double depth_alpha    = 0.7;  // 新深度权重，0.6~0.8 常用
    cv::Mat      cached_depth;
    bool         has_cached_depth = false;
    bool smooth_depth_merge = false;  // 是否平滑融合深度结果，开启后会将当前深度结果与上次结果按权重融合，减少深度抖动

    cv::Mat out_frame;

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
        cv::Mat result_depth;
        {
            ScopedTimer timer("Depth inference");
            bool        do_depth = (!has_cached_depth) || ((num_frames - 1) % depth_interval == 0);
            if (do_depth) {
                result_depth = depth_model.predict(img);
                if (has_cached_depth && smooth_depth_merge) {
                    cv::addWeighted(result_depth, depth_alpha, cached_depth, 1.0 - depth_alpha, 0.0, result_depth);
                }
                cached_depth     = result_depth;
                has_cached_depth = true;
            } else {
                result_depth = cached_depth;
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

        // 先只做绘制
        {
            ScopedTimer timer("Draw");
            for (int i = 0; i < output_stracks.size(); i++) {
                const std::vector<float> & tlwh = output_stracks[i].tlwh;
                if (tlwh[2] * tlwh[3] <= 20) {
                    continue;
                }

                int         class_id = output_stracks[i].class_id;
                cv::Scalar  s        = tracker.get_color(output_stracks[i].track_id);
                std::string label    = cv::format("%s #%d", vClassNames[class_id].c_str(), output_stracks[i].track_id);

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

        cv::Mat depth_vis;
        if (result_depth.size() != img.size()) {
            cv::resize(result_depth, depth_vis, img.size());
        } else {
            depth_vis = result_depth;
        }

        out_frame.create(img.rows + depth_vis.rows, img.cols, img.type());
        img.copyTo(out_frame(cv::Rect(0, 0, img.cols, img.rows)));
        depth_vis.copyTo(out_frame(cv::Rect(0, img.rows, img.cols, depth_vis.rows)));

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
    std::cout << "Engine Compute FPS: " << (total_us > 0 ? (num_frames * 1000000LL / total_us) : 0) << std::endl;
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
