# author: young
import cv2
import numpy as np
import camera_configs
import open3d as o3d

img = cv2.imread('/Refer/data/img_4.jpg')
img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_left = img[0:720, 0:1280] #高度480，长度640 行数480 列数640
img_right = img[0:720, 1280:2560]

img_left_rectified = cv2.remap(img_left, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
img_right_rectified = cv2.remap(img_right, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)

concat = cv2.hconcat([img_left_rectified, img_right_rectified])

imgL = cv2.cvtColor(img_left_rectified, cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(img_right_rectified, cv2.COLOR_BGR2GRAY)
num = 7
blockSize = 10
#numDisparities视差窗口，即最大视差值与最小视差值之差,窗口大小必须是16的整数倍，int型
#blockSize：SAD窗口大小，5~21之间为宜

S = cv2.StereoSGBM_create(minDisparity=0, numDisparities=16*num, blockSize=blockSize)
dis = S.compute(imgL, imgR)


dis_color = dis
dis_color = cv2.normalize(dis_color, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
dis_color = cv2.applyColorMap(dis_color, 2)
cv2.imshow("depth", dis_color)


output_points = np.zeros((1280*720, 6))


i = 0

b = 117.39
f = 1003
cx = 636
cy = 518

for row in range(dis.shape[0]):
    for col in range(dis.shape[1]):
        if(dis[row][col] != 0 and dis[row][col] != (-16) and dis[row][col]>1100 and dis[row][col]<1570):
            output_points[i][0] = 16*b*(col-cx)/dis[row][col]
            output_points[i][1] = 16*b*(row-cy)/dis[row][col]
            output_points[i][2] = 16*b*f/dis[row][col]
            output_points[i][3] = img_color[row][col][0]
            output_points[i][4] = img_color[row][col][1]
            output_points[i][5] = img_color[row][col][2]
            i = i + 1

        # if(i == 250372):
        #     print('row:'+str(row)+','+'col:'+str(col))

def creatp_output(vertices, filename):
    ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
    with open(filename, 'w') as f:
        f.write(ply_header%dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')

output_file = 'nb.ply'
creatp_output(output_points, output_file)
pcd = o3d.io.read_point_cloud(output_file)
o3d.visualization.draw_geometries([pcd])

cv2.waitKey(0)




