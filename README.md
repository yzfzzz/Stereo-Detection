<img src="https://yzfzzz.oss-cn-shenzhen.aliyuncs.com/image/dafafa.drawio%20(5).png" alt="dafafa.drawio (5)" style="zoom: 80%;" />

## 项目日志

- [x] 双目相机的标定和初始化（2022.7.3）
- [x] 运行BM、SGBM算法（2022.7.6）
- [x] 研究SGBM算法并得出良好的open3d模型（2022.7.15）
- [x] 实现双目测距（2022.7.27）
- [x] 双目相机测出Yolov5检测物体的距离（2022.7.29）
- [x] 视频帧率提高至6FPS（2022.7.30）
- [x] 使用C++重勾BM算法（2022.8.1）
- [x] 使用C++重构SGBM算法（2022.8.1）
- [x] 使用TensorRT、C++部署yolov5模型（2022.8.3）
- [x] 完成项目，帧率至少达到20FPS（2022.8.3）



## 环境说明

- 🔥Tensorrt 8.4
- 🚀Cuda 11.6.1 Cudnn 8.4.1
- Opencv 4.5.1
- Cmake 3.23.3
- Visual Studio 2017
- MX350，Windows10



## 文件说明

- 💼**BM、SGBM**算法均有C++和Python两个版本
- 📂**tensorrt**：模型部署文件，帧率为20fps
- 📁**yolov5-v6.1-pytorch-master**：未部署前的python代码文件，帧率为5fps
- **stereo_introduce**：双目摄像头基本资料
- 📒**双目视觉资料**：从双目相机的标定（Matlab）到sgbm生成深度图的图文教程
- **stereo_shot.py**：摄像头拍摄代码



## 教程

1. 🍔YOLOv5 Tensorrt Python/C++部署：https://www.bilibili.com/video/BV113411J7nk/?spm_id_from=333.788.recommend_more_video.-1&vd_source=97aec9e652524c83bb4f4b9481ee059e
2. 🍞Pytorch 搭建自己的YoloV5目标检测平台Bubbliiiing：https://www.bilibili.com/video/BV1FZ4y1m777?spm_id_from=333.999.0.0
3. 🍟双目摄像头-立体视觉：https://blog.csdn.net/qq_41204464/category_10766478.html?spm=1001.2014.3001.5482)
4. CUDA的正确安装/升级/重装/使用方式：https://zhuanlan.zhihu.com/p/520536351
5. 报错【Could not locate zlibwapi.dll. Please make sure it is in your library path】：https://blog.csdn.net/qq_44224801/article/details/125525721











