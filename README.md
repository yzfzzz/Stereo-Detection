![GitHub](./doc/product.png)

# Stereo-Detection C++ 部署框架

基于 C++ 和 TensorRT 的高性能视觉推理框架。它将 **YOLO 目标检测**、**单目/双目深度估计 (Depth-Anything / Lite-Mono)** 结合在一起，并利用基于 **卡尔曼滤波 (Kalman Filter)** 的平滑预估模块，实时计算目标的相对运动状态（靠近/远离，加速/减速）。

该框架原生支持在 x86 (例如 RTX 5060) 和 aarch64 (例如 Jetson Nano) 平台上部署，旨在提供端到端的极低延迟实时推理方案。

## 📁 目录结构

```text
├── cpp/
│   ├── CMakeLists.txt          # 顶层 CMake 构建脚本
│   ├── main.cpp                # 核心推理主程序
│   ├── bin/                    # 编译产出及执行目录
│   │   ├── config.yaml         # 全局配置文件
│   ├── core/                   # 核心基类与管理层 (ConfigManager, MotionStateEngine 等)
│   ├── depth/                  # 深度模型的 TensorRT 推理后端
│   ├── detect/                 # YOLO 检测模型的 TensorRT 推理后端
│   └── tools/                  # 辅助工具模块 (DrawingManager可视化, IOManager文件读写, Timer等)
├── data/                       # 测试音视频流与校准数据存放目录
├── doc/                        # 文档与算法原型资料
└── model/                      # 模型文件
```

## ✨ 核心特性

- **多模型流水协同**：高度集成了 YOLO 目标检测引擎与 Lite-Mono/Depth-Anything 深度估计模型。
- **零延迟卡尔曼平滑跟拍 (Kalman Filter)**：废弃了传统的滑动平均滤波(SMA)，以 OpenCV `KalmanFilter` 构建二阶运动模型。无滞后地输出目标瞬时速度、加速度、位置等动态参数。
- **克服 Depth-Bleeding (边缘深度穿透)**：采用了严苛的 YOLO NMS 抑制，和遮挡/截断边界判定，防止后景检测框吸取前景距离。
- **面向对象封装设计**：
  - `ConfigManager`：集成 `yaml-cpp`，分版块无缝挂载层级参数表。
  - `IOManager`：支持安全创建 OS 目录栈，提供（视频流 / 图片序列 / 纯日志）的组合保存模式。
  - `DrawingManager`：统筹处理 OpenCV GUI 文字标签渲染、状态染色及多视图拼接。

模块 / 环节    耗时    说明

- Cap read    5.72 ms    视频流 / 摄像头图像读取
- YOLO inference    61.89 ms    YOLO 目标检测推理
- ByteTrack    0.55 ms    目标跟踪算法处理
- Depth inference    108.18 ms    深度估计模型推理
- Write    132.20 ms    结果写盘 / 视频帧输出保存
- Draw    8.80 ms    检测框 / 深度图等可视化绘制
- One frame average time    319.34 ms    单帧处理总平均耗时（含所有环节）
- Infer Compute FPS    5 FPS    引擎端有效推理帧率

## 🚀 编译与构建

系统需要安装：**CMake (>= 3.10)**, **CUDA & cuDNN**, **TensorRT**, **OpenCV**, 以及 **yaml-cpp**。

```bash
cd cpp
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

编译成功后，可执行文件 `main` 会生成在 bin 目录下。


## ▶️ 运行项目

执行编译出的 `main` 二进制文件，并传入所需检测的视频绝对/相对路径：

```bash
cd cpp/bin
./main <video_path>
```

**示例：**
```bash
./main ../../data/test_video.mp4
```

运行后：终端将实时输出状态变化流。同时在 `ConfigManager` 指定的回调目录（如 `cpp/bin/out_dir/`）下保存包含目标运动高亮涂层的可视化结果。

## 📊 性能表现（参考）

- 在 RTX 5060 下使用 `YOLOv8` + Lite-Mono，整个推理流水线的模型显存总增量可以控制在 ~200MB 以内，FPS 稳定在 90+。
- 该显存占用亦极其契合显存受限的边缘设备（如 **Jetson Nano 2GB**），推荐在 Jetson 上使用 INT8/FP16 精度的引擎以获取更好的实时性表现。
