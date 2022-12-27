# 双目检测常见问题答疑

> **开源链接**：https://github.com/yzfzzz/Stereo-Detection 
>
> **CSDN**：https://blog.csdn.net/henghuizan2771

## 1.那个权重文件是在哪里?Yolov5学习资料？

https://www.bilibili.com/video/BV1FZ4y1m777/

![image-20221227113838569](https://yzfzzz.oss-cn-shenzhen.aliyuncs.com/image/image-20221227113838569.png)

## 2.双目摄像机的型号？在那买？

任意一款双目相机都可以，我的仅供参考：淘宝汇博视捷，80度无畸变，焦距3mm，基线12cm

| <img src="https://yzfzzz.oss-cn-shenzhen.aliyuncs.com/image/image-20221227113332551.png" alt="image-20221227113332551" style="zoom:130%;" /> | ![image-20221227113310358](https://yzfzzz.oss-cn-shenzhen.aliyuncs.com/image/image-20221227113310358.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

## 3.加上测距之后帧率变得这么低？

因为我们安装的opencv都是用CPU运算，所以sgbm算法主要是CPU在工作，而Jeston nano的CPU较差。导致严重拖慢速度。

🚀解决方法：安装opencv的GPU版本，然后在设备上编译，最后使用C++重构算法

## 4.如何改成相机实时测距、测距不想用视频测量而是用图片测量，该如何修改代码？

建议先学一下opencv的基础知识，基础不牢地动山摇！

## 5.相同的标定数据，sgbm匹配误差比bm小的多，比如实际距离1.6m，前者测1.7m，但是后者达到160m,差距这么大？跑出来识别的距离都很大，离得很近都显示有200多？

- sgbm算法比BM算法精度上高很多，这就造成了sgbm算法速度比BM更慢
- 跑出来识别的距离都很大，一般是因为没有计算出距离，然后随便报了一个很大的数字。建议检查一下标定过程，或者相机不能离被测物体太近（大于1米）

## 6.Jeston nano部署时yolov5的版本？

- 在Jeston nano部署的时候我采用官方的yolov5-6.0或yolov5-6.1版本
- 不部署时，我采用的是https://www.bilibili.com/video/BV1FZ4y1m777/的yolov5

二者的权重是不一样的