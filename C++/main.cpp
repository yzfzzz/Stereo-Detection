#include <iostream>
#include <string>
#include "infer.h"
#include "BYTETracker.h"
#
#include "depth_anything.h"


// 需要跟踪的类别，可以根据自己需求调整，筛选自己想要跟踪的对象的种类（以下对应COCO数据集类别索引）
std::vector<int>  trackClasses {0, 1, 2, 3, 5, 7};  // person, bicycle, car, motorcycle, bus, truck


bool isTrackingClass(int class_id){
	for (auto& c : trackClasses){
		if (class_id == c) return true;
	}
	return false;
}


int run(char* videoPath){
    // read video
    std::string inputVideoPath = std::string(videoPath);
    cv::VideoCapture cap(inputVideoPath);
    if ( !cap.isOpened() ) return 0;

    int img_w = cap.get(CAP_PROP_FRAME_WIDTH);
	int img_h = cap.get(CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(CAP_PROP_FPS);
    long nFrame = static_cast<long>(cap.get(CAP_PROP_FRAME_COUNT));
    cout << "Total frames: " << nFrame << endl;

    cv::VideoWriter writer("result.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(img_w, img_h* 2 ));

    // YOLOv8 predictor
    std::string trtFile = "../../model/engine/yolov8s_fp16.engine";
    YoloDetector detector(trtFile, 0, 0.45, 0.01);

    // Depth Anything predictor
    std::string depthTrtFile = "../../model/engine/depth_anything_v2_vits.engine";
    DepthAnything depth_model;
    Logger logger;
    depth_model.init(depthTrtFile, logger);

    // ByteTrack tracker
    BYTETracker tracker(fps, 30);

    cv::Mat img;
    int num_frames = 0;
    long long  total_us = 0;
    while (true){
        if ( !cap.read(img) ) break;
        num_frames++;
        if (num_frames % 20 == 0){
            cout << "Processing frame " << num_frames << " (" << (total_us > 0 ? (num_frames * 1000000LL / total_us) : 0) << " fps)" << endl;
        }
        if (img.empty()) break;

        auto start = std::chrono::system_clock::now();

        // yolo inference
        cv::Mat result_depth = depth_model.predict(img);
        std::vector<Detection> res = detector.inference(img);

        // yolo output format to bytetrack input format, and filter bbox by class id
        std::vector<Object> objects;
        for (size_t j = 0; j < res.size(); j++){
            float* bbox = res[j].bbox;
            float conf = res[j].conf;
            int classId = res[j].classId;

            if (isTrackingClass(classId)){
                cv::Rect_<float> rect(bbox[0], bbox[1], (bbox[2] - bbox[0]), (bbox[3] - bbox[1]));
                Object obj {rect, classId, conf};
                objects.push_back(obj);
            }
        }

        // track
        std::vector<STrack> output_stracks = tracker.update(objects);

        auto end = std::chrono::system_clock::now();
        total_us += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // 上半部分：检测跟踪画面
        for (int i = 0; i < output_stracks.size(); i++)
		{
			std::vector<float> tlwh = output_stracks[i].tlwh;
			// bool vertical = tlwh[2] / tlwh[3] > 1.6;
			// if (tlwh[2] * tlwh[3] > 20 && !vertical)
            if (tlwh[2] * tlwh[3] > 20)
			{
                int class_id = output_stracks[i].class_id;
				cv::Scalar s = tracker.get_color(output_stracks[i].track_id);
                std::string label = cv::format("%s #%d", vClassNames[class_id].c_str(), output_stracks[i].track_id);

                int baseLine = 0;
                cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseLine);
                cv::Rect rect_bg(cv::Point(tlwh[0], tlwh[1] - label_size.height - 8),
                                cv::Size(label_size.width + 8, label_size.height + 8));
                cv::rectangle(img, rect_bg, s, cv::FILLED); // 填充背景
                cv::putText(img, label, cv::Point(tlwh[0] + 4, tlwh[1] - 4),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 2, cv::LINE_AA);

                // 绘制检测框
                cv::rectangle(img, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
			}
		}
        int show_fps = (total_us > 0) ? (num_frames * 1000000LL / total_us) : 0;
        cv::putText(img, cv::format("frame: %d fps: %d num: %ld", num_frames, show_fps, output_stracks.size()),
                    cv::Point(0, 30), 0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

        writer.write(img);
        cv::Mat top_frame = img;
        // 下半部分：原图 + 深度图叠加
        cv::Mat bottom_frame;
        if (result_depth.size() != top_frame.size())
        {
            cv::resize(result_depth, bottom_frame, top_frame.size());
        }
        else
        {
            bottom_frame = result_depth;
        }
        // 上下拼接写入
        cv::Mat out_frame;
        if (top_frame.size() != bottom_frame.size()) {
            cv::resize(top_frame, top_frame, bottom_frame.size());
        }
        cv::vconcat(top_frame, bottom_frame, out_frame);
        writer.write(out_frame);

        // cv::imshow("img", img);
        // char c = waitKey(1);
        // if (c > 0) break;
    }

    cap.release();
    std::cout << "FPS: " << (total_us > 0 ? (num_frames * 1000000LL / total_us) : 0)  << std::endl;

    return 0;
}


int main(int argc, char* argv[]){
    if (argc != 2 )
    {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "Usage: ./main [video path]" << std::endl;
        std::cerr << "Example: ./main ./videos/demo.mp4" << std::endl;
        return -1;
    }

    return run(argv[1]);
}
