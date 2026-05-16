#include "infer.h"
#include "utils.h"

int run(char * imageDir) {
    // get image file names for inferencing
    std::vector<std::string> file_names;
    if (read_files_in_dir(imageDir, file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    // create detector, and load engine plan
    std::string  trtFile = "../onnx_model/yolov8s_fp16.engine";
    YoloDetector detector(trtFile);

    // inference
    for (long unsigned int i = 0; i < file_names.size(); i++) {
        std::string imagePath = std::string(imageDir) + "/" + file_names[i];
        cv::Mat     img       = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (img.empty()) {
            continue;
        }

        auto start = std::chrono::system_clock::now();

        std::vector<Detection> res = detector.inference(img);

        auto end  = std::chrono::system_clock::now();
        int  cost = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Image: " << file_names[i] << " cost: " << cost << " ms." << std::endl;

        // draw result on image
        YoloDetector::drawImage(img, res);

        cv::imwrite("_" + file_names[i], img);

        // std::cout << "Image: " << file_names[i] << " done." << std::endl;
    }

    return 0;
}

int main(int argc, char * argv[]) {
    if (argc != 2) {
        printf("This program need 1 argument\n");
        printf("Usage: ./main [image dir]\n");
        printf("Example: ./main ./images\n");
        return 1;
    }

    return run(argv[1]);
}
