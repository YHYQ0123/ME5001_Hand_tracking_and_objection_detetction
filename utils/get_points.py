import cv2

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
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    # 创建一个窗口并绑定鼠标回调函数
    cv2.namedWindow("Camera")
    cv2.setMouseCallback("Camera", mouse_callback)

    print("Click 4 points in the video feed. The program will exit after 4 clicks.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Exiting...")
                break

            # 显示当前帧
            cv2.imshow("Camera", frame)

            # 按 'q' 键手动退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Manual exit.")
                break

    except KeyboardInterrupt:
        print("Stream stopped manually.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Video stream stopped.")
        print("Clicked points:", clicked_points)

if __name__ == "__main__":
    main()
