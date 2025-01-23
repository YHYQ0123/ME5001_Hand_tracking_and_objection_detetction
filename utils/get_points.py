import cv2
import numpy as np
from pykinect2 import PyKinectRuntime
from pykinect2.PyKinectV2 import *

# 全局变量，用于存储点击的坐标点
clicked_points = []

def mouse_callback(event, x, y, flags, param):
    """鼠标回调函数，用于捕捉点击的坐标"""
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Point {len(clicked_points)}: ({x}, {y})")  # 打印当前点坐标

        # 当点击点达到 4 个时，关闭窗口
        if len(clicked_points) == 4:
            print("Captured 4 points. Exiting...")
            cv2.destroyAllWindows()

def main():
    # 初始化 PyKinectRuntime，仅获取 RGB 摄像头数据
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)

    # 创建一个窗口并绑定鼠标回调函数
    cv2.namedWindow("Kinect RGB Camera")
    cv2.setMouseCallback("Kinect RGB Camera", mouse_callback)

    print("Click 4 points in the video feed. The program will exit after 4 clicks.")

    try:
        while True:
            # 检查是否有新的 RGB 帧
            if kinect.has_new_color_frame():
                # 获取 RGB 帧数据并转换为 numpy 数组
                frame = kinect.get_last_color_frame()
                frame = frame.reshape((1080, 1920, 4))  # Kinect V2 RGB 分辨率为 1920x1080，4 通道
                frame = frame[:, :, :3]  # 去掉 Alpha 通道
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换颜色通道顺序

                # 显示当前帧
                cv2.imshow("Kinect RGB Camera", frame)

            # 按 'q' 键手动退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Manual exit.")
                break

    except KeyboardInterrupt:
        print("Stream stopped manually.")
    finally:
        kinect.close()
        cv2.destroyAllWindows()
        print("Kinect stream stopped.")
        print("Clicked points:", clicked_points)

if __name__ == "__main__":
    main()