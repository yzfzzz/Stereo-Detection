# author: young
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

size = (1280, 720)  # open windows size
# R1:左摄像机旋转矩阵, P1:左摄像机投影矩阵, Q:重投影矩阵
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R, T)

# 校正查找映射表,将原始图像和校正后的图像上的点一一对应起来
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)

print(Q)


