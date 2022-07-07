import cv2
import time

counter = 1
AUTO = True  # 自动拍照，或手动按s键拍照
INTERVAL = 2 # 自动拍照间隔
camera = cv2.VideoCapture(0)#也许你可能要capture两次
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)#设置分辨率
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)#
utc = time.time()
folder = "D:\GDEE\Project\Refer\data" # 拍照文件目录

def shot( frame):
    global counter
    path = folder +"\img_" + str(counter) + ".jpg"
    cv2.imwrite(path, frame)
    print("snapshot saved into: " + path)


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
