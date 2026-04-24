#ifndef PUBLIC_H
#define PUBLIC_H

#include <iostream>
#include <fstream>
#include <unistd.h>
#include <string.h>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <map>

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#define CHECK(call) check(call, __LINE__, __FILE__)

inline bool check(cudaError_t e, int iLine, const char *szFile)
{
    if (e != cudaSuccess)
    {
        std::cout << "CUDA runtime API error " << cudaGetErrorName(e) << " at line " << iLine << " in file " << szFile << std::endl;
        return false;
    }
    return true;
}

using namespace nvinfer1;


class Logger : public ILogger
{
public:
    Severity reportableSeverity;

    Logger(Severity severity = Severity::kINFO):
        reportableSeverity(severity) {}

    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity > reportableSeverity)
        {
            return;
        }
        switch (severity)
        {
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
__inline__ size_t dataTypeToSize(DataType dataType)
{
    switch ((int)dataType)
    {
    case int(DataType::kFLOAT):
        return 4;
    case int(DataType::kHALF):
        return 2;
    case int(DataType::kINT8):
        return 1;
    case int(DataType::kINT32):
        return 4;
    case int(DataType::kBOOL):
        return 1;
    default:
        return 4;
    }
}

// get the string of a TensorRT shape
__inline__ std::string shapeToString(nvinfer1::Dims dim)
{
    std::string output("(");
    if (dim.nbDims == 0)
    {
        return output + std::string(")");
    }
    for (int i = 0; i < dim.nbDims - 1; i++)
    {
        output += std::to_string(dim.d[i]) + std::string(", ");
    }
    output += std::to_string(dim.d[dim.nbDims - 1]) + std::string(")");
    return output;
}

// get the string of a TensorRT data type
__inline__ std::string dataTypeToString(DataType dataType)
{
    switch (dataType)
    {
    case DataType::kFLOAT:
        return std::string("FP32 ");
    case DataType::kHALF:
        return std::string("FP16 ");
    case DataType::kINT8:
        return std::string("INT8 ");
    case DataType::kINT32:
        return std::string("INT32");
    case DataType::kBOOL:
        return std::string("BOOL ");
    default:
        return std::string("Unknown");
    }
}

#endif  // PUBLIC_H
