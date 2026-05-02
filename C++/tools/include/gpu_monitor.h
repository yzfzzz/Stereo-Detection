#pragma once
#include <nvml.h>

#include <iostream>

class GpuMemoryMonitor {
  public:
    GpuMemoryMonitor(int deviceIndex = 0) : deviceIndex_(deviceIndex), initialized_(false) {
        nvmlReturn_t result = nvmlInit();
        if (result != NVML_SUCCESS) {
            std::cerr << "[NVML] Failed to initialize: " << nvmlErrorString(result) << std::endl;
            return;
        }
        result = nvmlDeviceGetHandleByIndex(deviceIndex_, &device_);
        if (result != NVML_SUCCESS) {
            std::cerr << "[NVML] Failed to get device handle: " << nvmlErrorString(result) << std::endl;
            nvmlShutdown();
            return;
        }
        initialized_ = true;
    }

    ~GpuMemoryMonitor() {
        if (initialized_) {
            nvmlShutdown();
        }
    }

    // 返回已用显存（字节）
    size_t GetUsedMemory() const {
        if (!initialized_) {
            return 0;
        }
        nvmlMemory_t memInfo;
        nvmlReturn_t result = nvmlDeviceGetMemoryInfo(device_, &memInfo);
        if (result != NVML_SUCCESS) {
            std::cerr << "[NVML] Failed to get memory info: " << nvmlErrorString(result) << std::endl;
            return 0;
        }
        return memInfo.used;
    }

    // 返回总显存（字节）
    size_t GetTotalMemory() const {
        if (!initialized_) {
            return 0;
        }
        nvmlMemory_t memInfo;
        nvmlReturn_t result = nvmlDeviceGetMemoryInfo(device_, &memInfo);
        if (result != NVML_SUCCESS) {
            std::cerr << "[NVML] Failed to get memory info: " << nvmlErrorString(result) << std::endl;
            return 0;
        }
        return memInfo.total;
    }

    // 打印显存占用
    void PrintMemoryUsage() const {
        if (!initialized_) {
            return;
        }
        nvmlMemory_t memInfo;
        nvmlReturn_t result = nvmlDeviceGetMemoryInfo(device_, &memInfo);
        if (result == NVML_SUCCESS) {
            std::cout << "[NVML] GPU Memory Used: " << memInfo.used / 1024.0 / 1024.0 << " MB / "
                      << memInfo.total / 1024.0 / 1024.0 << " MB" << std::endl;
        } else {
            std::cerr << "[NVML] Failed to get memory info: " << nvmlErrorString(result) << std::endl;
        }
    }

  private:
    int          deviceIndex_;
    nvmlDevice_t device_;
    bool         initialized_;
};
