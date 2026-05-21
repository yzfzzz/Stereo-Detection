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

#define CHECK_CUDA(call)                                                                                        \
    do {                                                                                                        \
        cudaError_t status = call;                                                                              \
        if (status != cudaSuccess) {                                                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(status) \
                      << std::endl;                                                                             \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
    } while (0)

using namespace nvinfer1;

class Logger : public ILogger {
  public:
    Severity reportableSeverity;

    Logger(Severity severity = Severity::kINFO) : reportableSeverity(severity) {}

    void log(Severity severity, const char * msg) noexcept override {
        if (severity > reportableSeverity) {
            return;
        }
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                std::cerr << "INTERNAL_ERROR: ";
                break;
            case Severity::kERROR:
                std::cerr << "ERROR: ";
                break;
            case Severity::kWARNING:
                std::cerr << "WARNING: ";
                break;
            case Severity::kINFO:
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

#endif  // PUBLIC_H
