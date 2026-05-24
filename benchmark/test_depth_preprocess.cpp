#include "config_manager.h"
#include "depth_anything.h"
#include "lite_mono.h"
#include "preprocess.h"

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <memory>
#include <opencv2/opencv.hpp>

class PreprocessBenchmark : public benchmark::Fixture {
  public:
    std::unique_ptr<BaseDepthModel> depth_model;
    cv::VideoCapture                cap;
    cv::Mat                         img;
    int                             num_frames = 0;

    // GPU buffers
    uchar * gpu_src  = nullptr;
    float * gpu_dst  = nullptr;
    float * mean_dev = nullptr;
    float * std_dev  = nullptr;

    int          raw_w = 0, raw_h = 0;
    int          model_w = 640, model_h = 192;
    cudaStream_t stream = 0;

    void SetUp(const ::benchmark::State & state) override {
        ConfigManager config_manager("config.yaml");

        cap.open("../data/shu/1shu_east_0514.mp4");
        if (!cap.isOpened()) {
            throw std::runtime_error("Failed to open video");
        }

        raw_w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        raw_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

        // 初始化模型并获取实际的 input size
        if (config_manager.getDepthEnginePath().find("depth_anything") != std::string::npos) {
            depth_model = std::make_unique<DepthAnything>();
        } else {
            depth_model = std::make_unique<LiteMono>();
        }
        depth_model->init(config_manager.getDepthEnginePath(), raw_w, raw_h);

        // 创建 CUDA stream
        cudaStreamCreate(&stream);
        cudaMalloc(&gpu_dst, 3 * model_h * model_w * sizeof(float));

        // Warmup: 读前 20 帧并预热
        for (int i = 0; i < 20; ++i) {
            if (cap.read(img) && !img.empty()) {
                // CPU warmup
                auto out = depth_model->preProcess(img);

                // GPU warmup
                depth_model->preProcessAsync(img);
                cudaStreamSynchronize(stream);
            }
        }

        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        num_frames = 0;
    }

    void TearDown(const ::benchmark::State & state) override {
        if (gpu_dst) {
            cudaFree(gpu_dst);
        }
        if (mean_dev) {
            cudaFree(mean_dev);
        }
        if (std_dev) {
            cudaFree(std_dev);
        }
        if (stream) {
            cudaStreamDestroy(stream);
        }
        depth_model.reset();
        cap.release();
    }
};

// CPU Preprocess + cudaMemcpy 到 buffer[0]
BENCHMARK_DEFINE_F(PreprocessBenchmark, CPU_Preprocess_With_Copy)(benchmark::State & state) {
    for (auto _ : state) {
        state.PauseTiming();
        if (!cap.read(img) || img.empty()) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }
        num_frames++;
        state.ResumeTiming();

        // CPU 前处理
        std::vector<float> input = depth_model->preProcess(img);

        // 同步拷贝到 GPU（与原来的 Predict 一致）
        cudaMemcpy(gpu_dst, input.data(), 3 * model_h * model_w * sizeof(float),
                   cudaMemcpyHostToDevice);

        benchmark::DoNotOptimize(gpu_dst);
    }
    state.SetItemsProcessed(state.iterations());
}

// GPU depthPreprocess（直接在 GPU 上做前处理）
BENCHMARK_DEFINE_F(PreprocessBenchmark, GPU_depthPreprocess)(benchmark::State & state) {
    for (auto _ : state) {
        state.PauseTiming();
        if (!cap.read(img) || img.empty()) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }
        num_frames++;
        state.ResumeTiming();

        depth_model->preProcessAsync(img);
        // 等待 kernel 完成
        cudaStreamSynchronize(stream);

        benchmark::DoNotOptimize(gpu_dst);
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_REGISTER_F(PreprocessBenchmark, CPU_Preprocess_With_Copy)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

BENCHMARK_REGISTER_F(PreprocessBenchmark, GPU_depthPreprocess)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

BENCHMARK_MAIN();
