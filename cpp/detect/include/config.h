#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <vector>

const int GPU_ID    = 0;
const int NUM_CLASS = 80;

const float NMS_THRESH  = 0.45f;
const float CONF_THRESH = 0.25f;
const int   MAX_NUM_OUTPUT_BBOX =
    1000;  // assume the box outputs no more than MAX_NUM_OUTPUT_BBOX boxes that
           // conf >= NMS_THRESH;
const int NUM_BOX_ELEMENT = 7;  // left, top, right, bottom, confidence, class,
                                // keepflag(whether drop when NMS)

const std::string ONNX_FILE = "../onnx_model/yolov8s.onnx";
// const std::string trtFile = "./yolov8s.plan";
// const std::string testDataDir = "../images";  // 用于推理

// for FP16 mode
const bool        B_FP16_MODE = false;
// for INT8 mode
const bool        B_INT8_MODE = false;
const std::string CACHE_FILE  = "./int8.cache";
const std::string CALIBRATION_DATA_PATH = "../calibrator";  // 存放用于 int8 量化校准的图像

const std::vector<std::string> V_CLASS_NAMES{ "person",        "bicycle",      "car",
                                              "motorcycle",    "airplane",     "bus",
                                              "train",         "truck",        "boat",
                                              "traffic light", "fire hydrant", "stop sign",
                                              "parking meter", "bench",        "bird",
                                              "cat",           "dog",          "horse",
                                              "sheep",         "cow",          "elephant",
                                              "bear",          "zebra",        "giraffe",
                                              "backpack",      "umbrella",     "handbag",
                                              "tie",           "suitcase",     "frisbee",
                                              "skis",          "snowboard",    "sports ball",
                                              "kite",          "baseball bat", "baseball glove",
                                              "skateboard",    "surfboard",    "tennis racket",
                                              "bottle",        "wine glass",   "cup",
                                              "fork",          "knife",        "spoon",
                                              "bowl",          "banana",       "apple",
                                              "sandwich",      "orange",       "broccoli",
                                              "carrot",        "hot dog",      "pizza",
                                              "donut",         "cake",         "chair",
                                              "couch",         "potted plant", "bed",
                                              "dining table",  "toilet",       "tv",
                                              "laptop",        "mouse",        "remote",
                                              "keyboard",      "cell phone",   "microwave",
                                              "oven",          "toaster",      "sink",
                                              "refrigerator",  "book",         "clock",
                                              "vase",          "scissors",     "teddy bear",
                                              "hair drier",    "toothbrush" };

#endif  // CONFIG_H
