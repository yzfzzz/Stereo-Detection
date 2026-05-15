![GitHub](./doc/product.png)

# Stereo-Detection C++ 部署框架

本项目是一个基于 C++ 和 TensorRT 的高性能多任务视觉推理框架。它将 **YOLO 目标检测**、**单目深度估计 (Depth-Anything / Lite-Mono)** 以及 **ByteTrack 多目标跟踪** 结合在一起，并能利用深度变化趋势估算目标的相对运动状态（靠近/远离、加速/减速）。

该框架原生支持在 x86 (例如 RTX 5060) 和 aarch64 (例如 Jetson Nano) 平台上部署，旨在提供端到端的极低延迟实时推理方案。

## 📁 目录结构

```text
C++/
├── CMakeLists.txt          # 顶层 CMake 构建脚本
├── main.cpp                # 核心推理串联主程序（音视频 IO、推理流水线、可视化）
├── bin/                    # 编译产出目录
│   └── config.yaml         # 各模型 TensorRT Engine 路径及运行配置
├── bytetrack/              # ByteTrack 多目标跟踪算法的 C++ 实现
├── core/                   # 核心计算与通用 CUDA Kernel (如前处理加速)
├── depth/                  # 深度估计模型（Depth-Anything & Lite-Mono）的 TensorRT 实现
├── detect/                 # YOLOv8 目标检测模型的 TensorRT 实现
└── tools/                  # 通用工具类组件（如 Logger、时间统计、显存监控 GpuMemoryMonitor 等）
```

## ✨ 核心特性

- **多模型并行协同**：整合了 YOLOv8、Depth-Anything 和 Lite-Mono 模型。
- **TensorRT 加速**：基于 `nvinfer1` API 开发，支持 FP32/FP16/INT8 等多精度引擎的高效推理。
- **目标运动趋势分析**：通过融合 ByteTrack 的跟踪 ID 与单目深度图，实时计算目标相对于相机的“速度趋势”和“加速度趋势”。
- **灵活热切换**：通过 `config.yaml` 灵活选择使用的深度模型（Depth-Anything 或 LiteMono）和权重路径。
- **全方位性能监控**：集成了 `ScopedTimer` 计算各个子模块的耗时，通过基于 NVML 的 `GpuMemoryMonitor` 精确跟踪显存峰值和增量开销。

模块 / 环节    耗时    说明

- Cap read    5.72 ms    视频流 / 摄像头图像读取
- YOLO inference    61.89 ms    YOLO 目标检测推理
- ByteTrack    0.55 ms    目标跟踪算法处理
- Depth inference    108.18 ms    深度估计模型推理
- Write    132.20 ms    结果写盘 / 视频帧输出保存
- Draw    8.80 ms    检测框 / 深度图等可视化绘制
- One frame average time    319.34 ms    单帧处理总平均耗时（含所有环节）
- Infer Compute FPS    5 FPS    引擎端有效推理帧率

## 📦 环境依赖

在编译与运行之前，请确保系统中已经安装了以下依赖：

- **C++14 / C++17** 兼容的编译器
- **CMake** (>= 3.10)
- **CUDA & cuDNN** 
- **TensorRT** (建议 8.x 及以上)
- **OpenCV** (含 OpenCV C++ 开发库)
- **yaml-cpp** (用于解析配置文件)
- **NVIDIA Management Library (NVML)** (用于显存监控监测，通常随 NVIDIA 驱动自带)

## 🚀 编译与构建

项目采用 CMake 进行构建：

```bash
cd C++
mkdir build && cd build
cmake ..
make -j$(nproc)
```

编译成功后，可执行文件 `main` 会生成在 bin 目录下。

## ⚙️ 配置说明

在运行前，请确认 `bin/config.yaml` 中的模型路径指向了正确的从 ONNX / PyTorch 转换来的 TensorRT `.engine` 文件：

```yaml
# bin/config.yaml 示例
yolo_engine: "../model/engine/yolov8s_fp16.engine"
depth_engine: "../model/engine/lite_mono_fp16.engine"
```

## ▶️ 运行项目

执行编译出的 `main` 二进制文件，并传入需要处理的视频路径即可：

```bash
cd ../bin
./main <video_path>
```

**例如：**

```bash
./main ../../data/3_car.mp4
```

运行过程中，终端会输出 `fps`、模块耗时及 `NVML` 的显存占用信息；同时程序会在当前目录生成一个带有可视化目标框、跟踪 ID 以及运动状态标签的 `result.mp4` 文件。

## 📊 性能表现（参考）

- 在 RTX 5060 下使用 `YOLOv8` + Lite-Mono，整个推理流水线的模型显存总增量可以控制在 ~200MB 以内，FPS 稳定在 90+。
- 该显存占用亦极其契合显存受限的边缘设备（如 **Jetson Nano 2GB**），推荐在 Jetson 上使用 INT8/FP16 精度的引擎以获取更好的实时性表现。
