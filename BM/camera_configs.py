# author: young
import cv2
import numpy as np

# 效果好
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

# 角度很多但效果一般
# right_camera_matrix = np.array([[1011.630992,6.392290621,667.5686089],
#                                [ 0,1013.460245,503.4011875],
#                                [0., 0,1.0000]])
# right_distortion = np.array([[-0.078598405,0.181429846, 0.005991071,0.011485758,-0.216528642]])
#
# left_camera_matrix = np.array([[999.2148594,1.517083305,664.099101],[ 0.,1004.928519,509.320943,], [0., 0,1.0000]])
# left_distortion = np.array([[-0.068650958,0.329526482,0.009413124,0.007593334, -0.762189196]])
#
# R = np.matrix([[ 0.999995654,-0.001219991,-0.002683778],[0.001249395,0.999938918,0.010981796],[ 0.002670216,-0.010985102,0.999936097],])
# T = np.array([-115.3587822,-0.643409169,1.336821271])




# 墙平移
# left_camera_matrix = np.array([[1023.60598,2.050151356,602.5534506],[0,976.0768203,398.6863484],[0.,0.,1.]])
#
# # left_distortion = np.array([[-0.154511565,0.325173292, 0.006934081,0.017466934, -0.340007548]])
# left_distortion = np.array([[-0.001241482,	0.372245099, -0.02528585,	0.00214508,-1.007572978]])
#
# right_camera_matrix = np.array([[1117.411713,-7.109583921,648.9929444],[0,1069.494628,404.4101094],[0.,0.,1.]])
#
# # right_distortion = np.array([[-0.192887524,0.706728768, 0.004233541,0.021340116,-1.175486913]])
# right_distortion = np.array([[0.064401477,0.162144856,-0.021187394,	0.019547213,-0.331058671]])
#
# R = np.array([[0.999965088,0.007987609,0.002453816], [-0.008017785,0.999889171,0.012544361],[-0.002353345,-0.012563598,0.999918306]])
#
# T = np.array([-181.8512309,	-19.42425901,125.5877407])


size = (1280, 720)  # open windows size
# R1:左摄像机旋转矩阵, P1:左摄像机投影矩阵, Q:重投影矩阵
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R, T)

# 校正查找映射表,将原始图像和校正后的图像上的点一一对应起来
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)

print(Q)

import numpy as np


####################仅仅是一个示例###################################


# 双目相机参数
class stereoCamera(object):
    def __init__(self):
        # 左相机内参
        self.cam_matrix_left = left_camera_matrix
        # 右相机内参
        self.cam_matrix_right = right_camera_matrix

        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = left_distortion
        self.distortion_r = right_distortion

        # 旋转矩阵
        self.R = R

        # 平移矩阵
        self.T = T

        # 焦距
        self.focal_length = 859.367  # 默认值，一般取立体校正后的重投影矩阵Q中的 Q[2,3]

        # 基线距离
        self.baseline = 119.9578  # 单位：mm， 为平移向量的第一个参数（取绝对值）







