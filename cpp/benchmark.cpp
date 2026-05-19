#include "config_manager.h"
#include "pipeline.h"

#include <benchmark/benchmark.h>

#include <opencv2/opencv.hpp>

class PipelineBenchmark : public benchmark::Fixture {
  public:
    std::unique_ptr<Pipeline> pipeline;
    cv::VideoCapture          cap;
    cv::Mat                   img;
    int                       num_frames = 0;

    void SetUp(const ::benchmark::State & state) override {
        ConfigManager config_manager("config.yaml");
        pipeline = std::make_unique<Pipeline>(config_manager);
        cap.open("../data/shu/1shu_east_0514.mp4");  // 替换为你的测试视频路径
        if (!cap.isOpened()) {
            throw std::runtime_error("Failed to open video");
        }
    }

    virtual void TearDown(benchmark::State & state) override {
        pipeline.reset();  // 显式释放 Pipeline，触发 YoloDetector 等的析构函数
        cap.release();
    }
};

BENCHMARK_DEFINE_F(PipelineBenchmark, ProcessInference)(benchmark::State & state) {
    for (auto _ : state) {
        if (!cap.read(img) || img.empty()) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);  // 视频播完自动重头
            continue;
        }
        num_frames++;
        FrameResult result;
        pipeline->process(img, num_frames, result);
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_REGISTER_F(PipelineBenchmark, ProcessInference)->Unit(benchmark::kMillisecond);
BENCHMARK_MAIN();
