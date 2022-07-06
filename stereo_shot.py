import cv2
import time

counter = 1
AUTO = True  # 自动拍照，或手动按s键拍照
INTERVAL = 2 # 自动拍照间隔
camera = cv2.VideoCapture(0)#也许你可能要capture两次
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)#设置分辨率
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)#
utc = time.time()
folder = "D:/DeepLearn/data" # 拍照文件目录

def shot( frame):
    global counter
    leftpath = folder +"/left/"+"left_" + str(counter) + ".jpg"
    rightpath=folder + "/right/"+ "right_" + str(counter) + ".jpg"
    leftframe=frame[0:720,0:1280]#这里是为了将合在一个窗口显示的图像分为左右摄像头
    rightframe=frame[0:720,1280:2560]
    cv2.imwrite(leftpath, leftframe)

    cv2.imwrite(rightpath, rightframe)
    print("snapshot saved into: " + leftpath)
    print("snapshot saved into: " + rightpath)

while True:
    ret, frame = camera.read()

    cv2.imshow("original", frame)

    now = time.time()
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("s"):
        shot( frame)
        counter += 1

camera.release()
cv2.destroyWindow("original")
