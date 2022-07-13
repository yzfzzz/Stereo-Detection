import cv2
import numpy as np

left_camera_matrix = np.array([[986.4572391,1.673607456,651.0717611],[0,1001.238398,535.8195077],[0.,0.,1.]])

# left_distortion = np.array([[-0.154511565,0.325173292, 0.006934081,0.017466934, -0.340007548]])
left_distortion = np.array([[-0.154511565,0.325173292, 0.006934081,0.017466934, 0]])

right_camera_matrix = np.array([[998.5848065,7.37746018,667.3698587],[0,1006.305891,528.9731771],[0.,0.,1.]])

# right_distortion = np.array([[-0.192887524,0.706728768, 0.004233541,0.021340116,-1.175486913]])
right_distortion = np.array([[-0.192887524,0.706728768, 0.004233541,0.021340116,0]])

R = np.array([[0.999925137,-0.003616734,-0.01168927],
              [0.003742452,0.999935202,0.010751105],
              [0.011649629,-0.010794046,0.999873879]])

T = np.array([-117.3364039,0.277054571,-3.7672413])
size = (1280, 720)

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)
# 校正查找映射表,将原始图像和校正后的图像上的点一一对应起来
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)

print(Q)
# -*- coding: utf-8 -*-

import numpy as np
import cv2
# import camera_configs
import random
import math

cap = cv2.VideoCapture(0)
cap.set(3, 2560)
cap.set(4, 720)  # 打开并设置摄像头


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


WIN_NAME = 'Deep disp'
cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)

folder = "D:\GDEE\Project\Refer\data" # 拍照文件目录


while True:
    ret, frame = cap.read()
    frame1 = frame[0:720, 0:1280]
    frame2 = frame[0:720, 1280:2560]  # 割开双目图像

    imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # 将BGR格式转换成灰度图片
    imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # cv2.remap 重映射，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程。
    # 依据MATLAB测量数据重建无畸变图片
    img1_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)

    imageL = cv2.cvtColor(img1_rectified, cv2.COLOR_GRAY2BGR)
    imageR = cv2.cvtColor(img2_rectified, cv2.COLOR_GRAY2BGR)

    # SGBM
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

    cv2.imshow("left", frame1)
    # cv2.imshow("right", frame2)
    # cv2.imshow("left_r", imgL)
    # cv2.imshow("right_r", imgR)
    cv2.imshow(WIN_NAME, disp)  # 显示深度图的双目画面

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    if key == ord('s'):
        cv2.imwrite("frame.jpg", frame)
        cv2.imwrite("disp.jpg", disp)


    # SGBM
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

numberOfDisparities = ((1280// 8) + 15) & -16  # 640对应是分辨率的宽

stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=numberOfDisparities, blockSize=9,
                               P1=8 * 1 * 9 * 9, P2=32 * 1 * 9 * 9, disp12MaxDiff=1, uniquenessRatio=10,
                               speckleWindowSize=100, speckleRange=32, mode=cv2.STEREO_SGBM_MODE_SGBM)

left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)