import cv2
import numpy as np
from pykinect2 import PyKinectRuntime
from pykinect2.PyKinectV2 import *

clicked_points = []

def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Point {len(clicked_points)}: ({x}, {y})")
        if len(clicked_points) == 4:
            print("Captured 4 points. Exiting...")
            cv2.destroyAllWindows()

def main():
    kinect = PyKinectRuntime.PyKinectRuntime(FrameSourceTypes_Color)
    
    cv2.namedWindow("Kinect RGB Camera")
    cv2.setMouseCallback("Kinect RGB Camera", mouse_callback)

    print("Click 4 points in the video feed. The program will exit after 4 clicks.")

    try:
        while True:
            if kinect.has_new_color_frame():
                # 获取原始颜色帧
                frame = kinect.get_last_color_frame()
                
                # 重塑帧的维度（使用Kinect的帧描述信息）
                frame = frame.reshape((kinect.color_frame_desc.Height, 
                                     kinect.color_frame_desc.Width, 4))
                
                # 转换为BGR格式并去除Alpha通道
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                # 裁剪中间区域（去除左右黑边）
                frame_cropped = frame_bgr[:, 440:1760]  # 裁剪宽度到1320像素
                
                # 调整到目标分辨率
                frame_resized = cv2.resize(frame_cropped, (660, 540))
                
                # 显示处理后的帧
                cv2.imshow("Kinect RGB Camera", frame_resized)

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