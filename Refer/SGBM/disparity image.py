# author: young
import cv2
import numpy as np
import camera_configs

img = cv2.imread('/Refer/data/img_4.jpg')

img_left = img[0:720, 0:1280] #高度480，长度640
img_right = img[0:720, 1280:2560]

cv2.imshow("original", img_left)
img_left_rectified = cv2.remap(img_left, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
img_right_rectified = cv2.remap(img_right, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)


imgL = cv2.cvtColor(img_left_rectified, cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(img_right_rectified, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('SGBM')
cv2.createTrackbar('num','SGBM',2,10, lambda x: None)
cv2.createTrackbar('blockSize','SGBM',5,255,lambda x: None) #在BM显示窗口创立滑条，在线调参

num = 0
blockSize = 0
app = 0
while(1):
    # 两个trackbar用来调节不同的参数查看效果
    num = cv2.getTrackbarPos('num', 'SGBM')
    blockSize = cv2.getTrackbarPos('blockSize', 'SGBM')
    if blockSize % 2 == 0:
        blockSize += 1
    if blockSize < 5:
        blockSize = 5
#numDisparities视差窗口，即最大视差值与最小视差值之差,窗口大小必须是16的整数倍，int型
#blockSize：SAD窗口大小，5~21之间为宜

    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=16*num, blockSize=blockSize)
    dis = stereo.compute(imgL, imgR)
    if(app==0):
        print("视差图维度："+str(dis.ndim))
        print(type(dis))
        max_index = np.unravel_index(np.argmax(dis, axis=None), dis.shape)
        app = 1

    # 计算出的视差是CV_16S格式-16位有符号整数（-32768…32767）
    dis = cv2.normalize(dis, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    dis = cv2.applyColorMap(dis, 2)
    cv2.imshow('SGBM', dis)

    cv2.waitKey(1000)





cv2.waitKey(0)
