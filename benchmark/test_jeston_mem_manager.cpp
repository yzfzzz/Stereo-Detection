#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#include <cstring>
#include <iostream>
#include <memory>

#define CHECK_CUDA(call)                                                                                        \
    do {                                                                                                        \
        cudaError_t status = call;                                                                              \
        if (status != cudaSuccess) {                                                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(status) \
                      << std::endl;                                                                             \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
    } while (0)

enum ResolutionKey { P480 = 0, P720 = 1, K1 = 2, K2 = 3, K4 = 4 };

static const std::map<std::string, std::pair<size_t, size_t>> kResolutionMap = {
    { "480P", { 854, 480 }   },
    { "720P", { 1280, 720 }  },
    { "1K",   { 1920, 1080 } },
    { "2K",   { 2560, 1440 } },
    { "4K",   { 3840, 2160 } }
};

static const std::vector<std::string> kResNames = { "480P", "720P", "1K", "2K", "4K" };

// 统一内存方式（Jetson 优化）
class JetsonManagedMemory {
  public:
    void allocate(size_t width, size_t height) {
        size_t depth_size    = width * height * sizeof(unsigned char);
        size_t colormap_size = width * height * sizeof(char) * 3;

#if defined(__aarch64__)
        CHECK_CUDA(cudaMallocManaged(&depth_buffer, depth_size));
        CHECK_CUDA(cudaMallocManaged(&colormap_buffer, colormap_size));
#else
        std::cerr << "Unified memory optimization is only enabled for ARM64 (Jetson) platforms." << std::endl;
        exit(EXIT_FAILURE);
#endif

        depth_output    = (unsigned char *) depth_buffer;
        colormap_output = (char *) colormap_buffer;
        this->width     = width;
        this->height    = height;
    }

    void transferHostToDevice(cudaStream_t stream) {
#if defined(__aarch64__)
        // 统一内存：使用 cudaMemcpy 触发 CPU -> GPU 页迁移
        CHECK_CUDA(cudaMemcpyAsync(depth_buffer, depth_output, width * height * sizeof(unsigned char),
                                   cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(colormap_buffer, colormap_output, width * height * sizeof(char) * 3,
                                   cudaMemcpyHostToDevice, stream));
#endif
    }

    void transferDeviceToHost(cudaStream_t stream) {
#if defined(__aarch64__)
        // 统一内存：使用 cudaMemcpy 触发 GPU -> CPU 页迁移
        CHECK_CUDA(cudaMemcpyAsync(depth_output, depth_buffer, width * height * sizeof(unsigned char),
                                   cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaMemcpyAsync(colormap_output, colormap_buffer, width * height * sizeof(char) * 3,
                                   cudaMemcpyDeviceToHost, stream));
#else
        std::cerr << "Unified memory optimization is only enabled for ARM64 (Jetson) platforms." << std::endl;
        exit(EXIT_FAILURE);
#endif
    }

    void cleanup() {
        if (depth_buffer) {
            CHECK_CUDA(cudaFree(depth_buffer));
        }
        if (colormap_buffer) {
            CHECK_CUDA(cudaFree(colormap_buffer));
        }
    }

    void *          depth_buffer    = nullptr;
    void *          colormap_buffer = nullptr;
    unsigned char * depth_output    = nullptr;
    char *          colormap_output = nullptr;
    size_t          width, height;
};

// 传统方式（离散显存 + memcpy）
class TraditionalMemory {
  public:
    void allocate(size_t width, size_t height) {
        size_t depth_size    = width * height * sizeof(unsigned char);
        size_t colormap_size = width * height * sizeof(char) * 3;

        CHECK_CUDA(cudaMalloc(&depth_buffer, depth_size));
        CHECK_CUDA(cudaMalloc(&colormap_buffer, colormap_size));

        depth_output    = new unsigned char[width * height];
        colormap_output = new char[width * height * 3];

        this->width  = width;
        this->height = height;
    }

    void transferHostToDevice(cudaStream_t stream) {
        CHECK_CUDA(cudaMemcpyAsync(depth_buffer, depth_output, width * height * sizeof(unsigned char),
                                   cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(colormap_buffer, colormap_output, width * height * sizeof(char) * 3,
                                   cudaMemcpyHostToDevice, stream));
    }

    void transferDeviceToHost(cudaStream_t stream) {
        CHECK_CUDA(cudaMemcpyAsync(depth_output, depth_buffer, width * height * sizeof(unsigned char),
                                   cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaMemcpyAsync(colormap_output, colormap_buffer, width * height * sizeof(char) * 3,
                                   cudaMemcpyDeviceToHost, stream));
    }

    void cleanup() {
        if (depth_buffer) {
            CHECK_CUDA(cudaFree(depth_buffer));
        }
        if (colormap_buffer) {
            CHECK_CUDA(cudaFree(colormap_buffer));
        }
        if (depth_output) {
            delete[] depth_output;
        }
        if (colormap_output) {
            delete[] colormap_output;
        }
    }

    void *          depth_buffer    = nullptr;
    void *          colormap_buffer = nullptr;
    unsigned char * depth_output    = nullptr;
    char *          colormap_output = nullptr;
    size_t          width, height;
};

// ==================== Jetson 统一内存 Benchmark ====================
class JetsonManagedMemoryBenchmark : public benchmark::Fixture {
  public:
    std::unique_ptr<JetsonManagedMemory> mem;
    cudaStream_t                         stream;
    size_t                               img_width  = 1920;
    size_t                               img_height = 1080;

    void SetUp(const ::benchmark::State & state) override {
        CHECK_CUDA(cudaStreamCreate(&stream));

        // 从参数中获取分辨率名称并查找对应宽高
        int res_idx = static_cast<int>(state.range(0));
        if (res_idx >= 0 && res_idx < kResNames.size()) {
            std::string res_name = kResNames[res_idx];
            auto        it       = kResolutionMap.find(res_name);
            if (it != kResolutionMap.end()) {
                img_width  = it->second.first;
                img_height = it->second.second;
            }
        } else {
            std::cerr << "Unsupported resolution index: " << res_idx << std::endl;
            exit(EXIT_FAILURE);
        }

        mem = std::make_unique<JetsonManagedMemory>();
        mem->allocate(img_width, img_height);

        // 初始化 Host 端数据源，用于 H2D 传输
        memset(mem->depth_output, 0xAA, img_width * img_height * sizeof(unsigned char));
        memset(mem->colormap_output, 0xBB, img_width * img_height * sizeof(char) * 3);

        // Warmup: 预热双向传输
        for (int i = 0; i < 5; i++) {
            mem->transferHostToDevice(stream);
            CHECK_CUDA(cudaStreamSynchronize(stream));
            mem->transferDeviceToHost(stream);
            CHECK_CUDA(cudaStreamSynchronize(stream));
        }
    }

    void TearDown(const ::benchmark::State & state) override {
        mem->cleanup();
        CHECK_CUDA(cudaStreamDestroy(stream));
    }
};

BENCHMARK_DEFINE_F(JetsonManagedMemoryBenchmark, HostToDeviceTransfer)(benchmark::State & state) {
    state.SetLabel(kResNames[static_cast<int>(state.range(0))]);
    for (auto _ : state) {
        mem->transferHostToDevice(stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_REGISTER_F(JetsonManagedMemoryBenchmark, HostToDeviceTransfer)
    ->Unit(benchmark::kMillisecond)
    ->Args({ P480 })
    ->Args({ P720 })
    ->Args({ K1 })
    ->Args({ K2 })
    ->Args({ K4 })
    ->Iterations(100);

BENCHMARK_DEFINE_F(JetsonManagedMemoryBenchmark, DeviceToHostTransfer)(benchmark::State & state) {
    state.SetLabel(kResNames[static_cast<int>(state.range(0))]);
    for (auto _ : state) {
        mem->transferDeviceToHost(stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_REGISTER_F(JetsonManagedMemoryBenchmark, DeviceToHostTransfer)
    ->Unit(benchmark::kMillisecond)
    ->Args({ P480 })
    ->Args({ P720 })
    ->Args({ K1 })
    ->Args({ K2 })
    ->Args({ K4 })
    ->Iterations(100);

// ==================== 传统内存 Benchmark ====================
class TraditionalMemoryBenchmark : public benchmark::Fixture {
  public:
    std::unique_ptr<TraditionalMemory> mem;
    cudaStream_t                       stream;
    size_t                             img_width  = 1920;
    size_t                             img_height = 1080;

    void SetUp(const ::benchmark::State & state) override {
        CHECK_CUDA(cudaStreamCreate(&stream));
        // 从参数中获取分辨率名称并查找对应宽高
        int res_idx = static_cast<int>(state.range(0));
        if (res_idx >= 0 && res_idx < kResNames.size()) {
            std::string res_name = kResNames[res_idx];
            auto        it       = kResolutionMap.find(res_name);
            if (it != kResolutionMap.end()) {
                img_width  = it->second.first;
                img_height = it->second.second;
            }
        } else {
            std::cerr << "Unsupported resolution index: " << res_idx << std::endl;
            exit(EXIT_FAILURE);
        }

        mem = std::make_unique<TraditionalMemory>();
        mem->allocate(img_width, img_height);

        // 初始化 Host 端数据源，用于 H2D 传输
        memset(mem->depth_output, 0xAA, img_width * img_height * sizeof(unsigned char));
        memset(mem->colormap_output, 0xBB, img_width * img_height * sizeof(char) * 3);

        // Warmup: 预热双向传输
        for (int i = 0; i < 5; i++) {
            mem->transferHostToDevice(stream);
            CHECK_CUDA(cudaStreamSynchronize(stream));
            mem->transferDeviceToHost(stream);
            CHECK_CUDA(cudaStreamSynchronize(stream));
        }
    }

    void TearDown(const ::benchmark::State & state) override {
        mem->cleanup();
        CHECK_CUDA(cudaStreamDestroy(stream));
    }
};

BENCHMARK_DEFINE_F(TraditionalMemoryBenchmark, HostToDeviceTransfer)(benchmark::State & state) {
    state.SetLabel(kResNames[static_cast<int>(state.range(0))]);
    for (auto _ : state) {
        mem->transferHostToDevice(stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_REGISTER_F(TraditionalMemoryBenchmark, HostToDeviceTransfer)
    ->Unit(benchmark::kMillisecond)
    ->Args({ P480 })
    ->Args({ P720 })
    ->Args({ K1 })
    ->Args({ K2 })
    ->Args({ K4 })
    ->Iterations(100);

BENCHMARK_DEFINE_F(TraditionalMemoryBenchmark, DeviceToHostTransfer)(benchmark::State & state) {
    state.SetLabel(kResNames[static_cast<int>(state.range(0))]);
    for (auto _ : state) {
        mem->transferDeviceToHost(stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK_REGISTER_F(TraditionalMemoryBenchmark, DeviceToHostTransfer)
    ->Unit(benchmark::kMillisecond)
    ->Args({ P480 })
    ->Args({ P720 })
    ->Args({ K1 })
    ->Args({ K2 })
    ->Args({ K4 })
    ->Iterations(100);

BENCHMARK_MAIN();
