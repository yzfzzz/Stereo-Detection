#ifndef PUBLIC_H
#define PUBLIC_H

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <string.h>
#include <unistd.h>

#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#define CHECK_CUDA(call)                                                          \
    do {                                                                          \
        cudaError_t status = call;                                                \
        if (status != cudaSuccess) {                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(status) << std::endl;                 \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

class Logger : public nvinfer1::ILogger {
  public:
    nvinfer1::ILogger::Severity reportable_severity_;

    Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO) :
        reportable_severity_(severity) {}

    void log(nvinfer1::ILogger::Severity severity, const char * msg) noexcept override {
        if (severity > reportable_severity_) {
            return;
        }
        switch (severity) {
            case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
                std::cerr << "INTERNAL_ERROR: ";
                break;
            case nvinfer1::ILogger::Severity::kERROR:
                std::cerr << "ERROR: ";
                break;
            case nvinfer1::ILogger::Severity::kWARNING:
                std::cerr << "WARNING: ";
                break;
            case nvinfer1::ILogger::Severity::kINFO:
                std::cerr << "INFO: ";
                break;
            default:
                std::cerr << "VERBOSE: ";
                break;
        }
        std::cerr << msg << std::endl;
    }
};

// get the size in byte of a TensorRT data type
__inline__ size_t dataTypeToSize(nvinfer1::DataType dataType) {
    switch ((int) dataType) {
        case int(nvinfer1::DataType::kFLOAT):
            return 4;
        case int(nvinfer1::DataType::kHALF):
            return 2;
        case int(nvinfer1::DataType::kINT8):
            return 1;
        case int(nvinfer1::DataType::kINT32):
            return 4;
        case int(nvinfer1::DataType::kBOOL):
            return 1;
        default:
            return 4;
    }
}

// get the string of a TensorRT shape
__inline__ std::string shapeToString(nvinfer1::Dims dim) {
    std::string output("(");
    if (dim.nbDims == 0) {
        return output + std::string(")");
    }
    for (int i = 0; i < dim.nbDims - 1; i++) {
        output += std::to_string(dim.d[i]) + std::string(", ");
    }
    output += std::to_string(dim.d[dim.nbDims - 1]) + std::string(")");
    return output;
}

// get the string of a TensorRT data type
__inline__ std::string dataTypeToString(nvinfer1::DataType dataType) {
    switch (dataType) {
        case nvinfer1::DataType::kFLOAT:
            return std::string("FP32 ");
        case nvinfer1::DataType::kHALF:
            return std::string("FP16 ");
        case nvinfer1::DataType::kINT8:
            return std::string("INT8 ");
        case nvinfer1::DataType::kINT32:
            return std::string("INT32");
        case nvinfer1::DataType::kBOOL:
            return std::string("BOOL ");
        default:
            return std::string("Unknown");
    }
}

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

#endif  // PUBLIC_H
