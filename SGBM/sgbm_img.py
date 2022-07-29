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
import numpy as np
import cv2
# import camera_configs
import random
import math

WIN_NAME = 'Deep disp'
cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)

frame = cv2.imread("yojuh-wewwn-004.jpg")
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
print(disparity)

disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # 归一化函数算法

dis_color = disparity
dis_color = cv2.normalize(dis_color, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
dis_color = cv2.applyColorMap(dis_color, 2)
cv2.imshow("depth", dis_color)
#
# cv2.imshow("left", frame1)
cv2.imshow(WIN_NAME, disp)  # 显示深度图的双目画面
cv2.waitKey()
# 销毁内存
cv2.destroyAllWindows()
