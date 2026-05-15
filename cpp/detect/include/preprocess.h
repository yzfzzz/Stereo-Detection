#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

void preprocess(const cv::Mat &srcImg, float *dstDevData, const int dstHeight,
                const int dstWidth, cudaStream_t stream);
/*
srcImg:     source image for inference
dstDevData: data after preprocess (resize / bgr to rgb / hwc to chw / normalize)
dstHeight:  CNN input height
dstWidth:   CNN input width
*/

#endif // PREPROCESS_H
