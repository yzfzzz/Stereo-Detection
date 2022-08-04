import cv2
import numpy as np

# 左镜头的内参，如焦距
left_camera_matrix = np.array([[516.5066236,-1.444673028,320.2950423],[0,516.5816117,270.7881873],[0.,0.,1.]])
right_camera_matrix = np.array([[511.8428182,1.295112628,317.310253],[0,513.0748795,269.5885026],[0.,0.,1.]])


# 畸变系数,K1、K2、K3为径向畸变,P1、P2为切向畸变
# left_distortion = np.array([[-0.154511565,0.325173292, 0.006934081,0.017466934, -0.340007548]])
left_distortion = np.array([[-0.046645194,0.077595167, 0.012476819,-0.000711358,0]])
# right_distortion = np.array([[-0.192887524,0.706728768, 0.004233541,0.021340116,-1.175486913]])
right_distortion = np.array([[-0.061588946,0.122384376,0.011081232,-0.000750439,0]])

# 旋转矩阵
R = np.array([[0.999911333,-0.004351508,0.012585312],
              [0.004184066,0.999902792,0.013300386],
              [-0.012641965,-0.013246549,0.999832341]])
# 平移矩阵
T = np.array([-120.3559901,-0.188953775,-0.662073075])
size = (640, 480)

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)
# 校正查找映射表,将原始图像和校正后的图像上的点一一对应起来
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)

print(Q)
# -*- coding: utf-8 -*-
# 鼠标回调函数
def onmouse_pick_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        threeD = param
        print('\n像素坐标 x = %d, y = %d' % (x, y))
        # print("世界坐标是：", threeD[y][x][0], threeD[y][x][1], threeD[y][x][2], "mm")
        print("世界坐标xyz 是：", threeD[y][x][0] / 1000.0, threeD[y][x][1] / 1000.0, threeD[y][x][2] / 1000.0, "m")

        distance = math.sqrt(threeD[y][x][0] ** 2 + threeD[y][x][1] ** 2 + threeD[y][x][2] ** 2)
        distance = distance / 1000.0  # mm -> m
        print("距离是：", distance, "m")


import numpy as np
# import camera_configs
import random
import math

# 2 加载视频文件
capture = cv2.VideoCapture("output.avi")
WIN_NAME = 'Deep disp'
cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)

# 3 读取视频
ret, frame = capture.read()
while ret:
    # 4 ret 是否读取到了帧，读取到了则为True
    cv2.imshow("video", frame)
    ret, frame = capture.read()
    img_color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame1 = frame[0:480, 0:640]
    frame2 = frame[0:480, 640:1280]  # 割开双目图像

    imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # 将BGR格式转换成灰度图片
    imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # cv2.remap 重映射，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程。
    # 依据MATLAB测量数据重建无畸变图片
    img1_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)

    imageL = cv2.cvtColor(img1_rectified, cv2.COLOR_GRAY2BGR)
    imageR = cv2.cvtColor(img2_rectified, cv2.COLOR_GRAY2BGR)

    cv2.imshow("img-L", imageL)
    cv2.imshow("img-R", imageR)

    # SGBM-室外
    blockSize = 8
    img_channels = 3
    stereo = cv2.StereoSGBM_create(minDisparity=1,
                                   numDisparities=64,
                                   blockSize=blockSize,
                                   P1=8 * img_channels * blockSize * blockSize,
                                   P2=32 * img_channels * blockSize * blockSize,
                                   disp12MaxDiff=-1,
                                   preFilterCap=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=100,
                                   mode=cv2.STEREO_SGBM_MODE_HH)

    disparity = stereo.compute(img1_rectified, img2_rectified)  # 计算视差

    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # 归一化函数算法

    threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)  # 计算三维坐标数据值
    threeD = threeD * 16

    # threeD[y][x] x:0~640; y:0~480;   !!!!!!!!!!
    cv2.setMouseCallback(WIN_NAME, onmouse_pick_points, threeD)  # 鼠标回调事件

    #
    # # 三维重建代码
    # import open3d as o3d
    #
    dis_color = disparity
    dis_color = cv2.normalize(dis_color, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    dis_color = cv2.applyColorMap(dis_color, 2)
    cv2.imshow("depth", dis_color)
    cv2.imshow("left", frame1)
    cv2.imshow(WIN_NAME, disp)  # 显示深度图的双目画面
    # 5 若键盘按下q则退出播放
    if cv2.waitKey(20) & 0xff == ord('q'):
        break

# 4 释放资源
capture.release()

# 5 关闭所有窗口
cv2.destroyAllWindows()