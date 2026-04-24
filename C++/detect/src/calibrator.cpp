#include <iostream>
#include <fstream>
#include <iterator>
#include <opencv2/opencv.hpp>
#include "calibrator.h"
#include "utils.h"

using namespace nvinfer1;


std::vector<float> preprocess(cv::Mat& img, int input_w, int input_h)
{
    int elements = 3 * input_h * input_w;

    // letterbox and resize
    int w, h, x, y;
    float r_w = input_w / (img.cols * 1.0);
    float r_h = input_h / (img.rows * 1.0);
    if (r_h > r_w){
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    }
    else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

    // HWC to CHW , BGR to RGB, Normalize
    std::vector<float> result(elements);
    float* norm_data = result.data();  // normalized data
    uchar* uc_pixel = out.data;
    for (int i = 0; i < input_h * input_w; i++)
    {
        norm_data[i] = (float)uc_pixel[2] / 255.0;
        norm_data[i + input_h * input_w] = (float)uc_pixel[1] / 255.0;
        norm_data[i + 2 * input_h * input_w] = (float)uc_pixel[0] / 255.0;
        uc_pixel += 3;
    }

    return result;
}


Int8EntropyCalibrator2::Int8EntropyCalibrator2(int batch_size, int input_w, int input_h, const char* img_dir, const char* calib_table_name, bool read_cache)
    : batch_size_(batch_size)
    , input_w_(input_w)
    , input_h_(input_h)
    , img_idx_(0)
    , img_dir_(img_dir)
    , calib_table_name_(calib_table_name)
    , read_cache_(read_cache)
{
    input_count_ = 3 * input_w * input_h * batch_size;
    // allocate memory for a batch of data, batchData is for CPU, deviceInput is for GPU
    batch_data = new float[input_count_];
    cudaMalloc(&device_input_, input_count_ * sizeof(float));
    read_files_in_dir(img_dir, img_files_);
}

Int8EntropyCalibrator2::~Int8EntropyCalibrator2()
{
    cudaFree(device_input_);
    if (batch_data) {
        delete[] batch_data;
    }
}

int Int8EntropyCalibrator2::getBatchSize() const noexcept
{
    return batch_size_;
}

bool Int8EntropyCalibrator2::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept
{
    if (img_idx_ + batch_size_ > (int)img_files_.size()) { return false; }

    float *ptr = batch_data;
    for (int i = img_idx_; i < img_idx_ + batch_size_; i++)
    {
        std::cout << img_files_[i] << "  " << i << std::endl;
        cv::Mat temp = cv::imread(img_dir_ + "/" + img_files_[i], cv::IMREAD_COLOR);
        if (temp.empty()){
            std::cerr << "Fatal error: image cannot open!" << std::endl;
            return false;
        }
        std::vector<float> input_data = preprocess(temp, input_w_, input_h_);
        memcpy(ptr, input_data.data(), (int)(input_data.size()) * sizeof(float));
        ptr += input_data.size();
    }
    img_idx_ += batch_size_;

    cudaMemcpy(device_input_, batch_data, input_count_ * sizeof(float), cudaMemcpyHostToDevice);
    bindings[0] = device_input_;
    return true;
}

const void* Int8EntropyCalibrator2::readCalibrationCache(size_t& length) noexcept
{
    std::cout << "reading calib cache: " << calib_table_name_ << std::endl;
    calib_cache_.clear();
    std::ifstream input(calib_table_name_, std::ios::binary);
    input >> std::noskipws;
    if (read_cache_ && input.good())
    {
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(calib_cache_));
    }
    length = calib_cache_.size();
    return length ? calib_cache_.data() : nullptr;
}

void Int8EntropyCalibrator2::writeCalibrationCache(const void* cache, size_t length) noexcept
{
    std::cout << "writing calib cache: " << calib_table_name_ << " size: " << length << std::endl;
    std::ofstream output(calib_table_name_, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
}
