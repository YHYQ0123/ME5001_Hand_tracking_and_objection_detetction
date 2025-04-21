import os
import cv2 as cv
import numpy as np
import time
from pykinect2 import PyKinectRuntime, PyKinectV2
import mediapipe as mp
from threading import Thread, Lock
from queue import Queue

# 初始化 Mediapipe 手部检测模块（降低检测复杂度）
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,  # 降低置信度阈值
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# 全局缓冲队列和锁
frame_queue = Queue(maxsize=30)
io_lock = Lock()

def detect_hand_keypoints_mediapipe(bgr_img):
    """优化版手部检测（缩小输入图像）"""
    # 先缩小图像再检测（降低计算量）
    small_img = cv.resize(bgr_img, (256, 144))
    rgb_img = cv.cvtColor(small_img, cv.COLOR_BGR2RGB)
    results = hands_detector.process(rgb_img)
    
    if not results.multi_hand_landmarks:
        return None
    
    # 恢复坐标到原始分辨率
    h, w, _ = bgr_img.shape
    hand_landmarks = results.multi_hand_landmarks[0]
    keypoints = []
    for lm in hand_landmarks.landmark:
        x_px = lm.x * 256 * (w / 256)
        y_px = lm.y * 144 * (h / 144)
        keypoints.append([x_px, y_px])
    return np.array(keypoints, dtype=np.float32)

def save_worker():
    """独立线程的保存任务"""
    while True:
        data = frame_queue.get()
        if data is None:
            break
            
        output_dir, frame_count, kinect_color, cam1_frame, cam2_frame, depth_disp, keypoints_3d = data
        
        # 并行保存图像
        def save_image(path, img):
            cv.imwrite(os.path.join(output_dir, path, f"{frame_count:04d}.jpg"), 
                       img, [cv.IMWRITE_JPEG_QUALITY, 85])
            
        threads = [
            Thread(target=save_image, args=("kinect_color", kinect_color)),
            Thread(target=save_image, args=("left_cam", cam1_frame)),
            Thread(target=save_image, args=("right_cam", cam2_frame)),
            Thread(target=save_image, args=("depth", depth_disp))
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # 记录关键点数据
        with io_lock:
            with open(os.path.join(output_dir, "hand_keypoints.dat"), "a") as f:
                if keypoints_3d is not None:
                    f.write(f"{frame_count:04d}.jpg " + " ".join(map(str, keypoints_3d)) + "\n")
                else:
                    f.write(f"{frame_count:04d}.jpg\n")
        frame_queue.task_done()

def capture_single_sequence(output_dir, kinect, cap1, cap2, disp_width, disp_height):
    """每次固定捕获20帧"""
    # 创建保存目录
    os.makedirs(os.path.join(output_dir, "kinect_color"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "left_cam"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "right_cam"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)
    
    # 清空关键点文件
    with open(os.path.join(output_dir, "hand_keypoints.dat"), "w") as f:
        pass

    frame_count = 0

    while frame_count < 20:
        # 优化 Kinect 帧处理
        color_frame = None
        while kinect.has_new_color_frame():
            color_frame = kinect.get_last_color_frame()
        if color_frame is not None:
            color_frame = color_frame.reshape((1080, 1920, 4))
            kinect_color = np.ascontiguousarray(color_frame[:, :, :3])
            kinect_color = cv.rotate(kinect_color, cv.ROTATE_180)
            kinect_color = cv.flip(kinect_color, 1)
        else:
            continue

        # 优化深度帧处理（降低分辨率）
        depth_frame = None
        while kinect.has_new_depth_frame():
            depth_frame = kinect.get_last_depth_frame()
        if depth_frame is not None:
            depth_frame = depth_frame.reshape((424, 512))
            kinect_depth_resized = cv.resize(depth_frame, (disp_width//2, disp_height//2))
            kinect_depth_resized = cv.rotate(kinect_depth_resized, cv.ROTATE_180)
            kinect_depth_resized = cv.flip(kinect_depth_resized, 1)
            
            # 快速深度可视化
            depth_display = cv.normalize(depth_frame, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
            depth_display = cv.applyColorMap(depth_display, cv.COLORMAP_JET)
            kinect_depth_disp = cv.resize(depth_display, (disp_width//2, disp_height//2))
            kinect_depth_disp = cv.rotate(kinect_depth_disp, cv.ROTATE_180)
            kinect_depth_disp = cv.flip(kinect_depth_disp, 1)
        else:
            continue

        # 优化摄像头捕获（降低分辨率）
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            continue
        cam1_frame = cv.resize(frame1, (disp_width//2, disp_height//2))
        cam2_frame = cv.resize(frame2, (disp_width//2, disp_height//2))

        # 手部关键点检测（优化版）
        keypoints_2d = detect_hand_keypoints_mediapipe(kinect_color)
        keypoints_3d = []
        if keypoints_2d is not None:
            for pt in keypoints_2d:
                x = int(pt[0] * (disp_width//2) / disp_width)
                y = int(pt[1] * (disp_height//2) / disp_height)
                x = max(0, min(x, (disp_width//2)-1))
                y = max(0, min(y, (disp_height//2)-1))
                depth_val = float(kinect_depth_resized[y, x])
                keypoints_3d.extend([pt[0], pt[1], depth_val])
        else:
            keypoints_3d = None

        # 使用多线程异步保存
        if not frame_queue.full():
            frame_queue.put((
                output_dir,
                frame_count,
                cv.resize(kinect_color, (disp_width//2, disp_height//2)),  # 降低保存分辨率
                cam1_frame,
                cam2_frame,
                kinect_depth_disp,
                keypoints_3d
            ))
            frame_count += 1

        # 显示低分辨率预览
        if cv.waitKey(1) & 0xFF == 27:
            break

    # 等待所有保存任务完成
    frame_queue.join()

def main_automated_capture():
    # 降低分辨率
    disp_width, disp_height = 960, 540
    total_motions = 16
    trials_per_motion = 5
    wait_time = 5  # 每次捕获后等待 5 秒

    # 初始化设备（优化参数）
    kinect = PyKinectRuntime.PyKinectRuntime(
        PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)
    cap1 = cv.VideoCapture(2)
    cap2 = cv.VideoCapture(1)
    cap1.set(cv.CAP_PROP_FRAME_WIDTH, disp_width)
    cap1.set(cv.CAP_PROP_FRAME_HEIGHT, disp_height)
    cap2.set(cv.CAP_PROP_FRAME_WIDTH, disp_width)
    cap2.set(cv.CAP_PROP_FRAME_HEIGHT, disp_height)
    
    # 优化摄像头设置
    cap1.set(cv.CAP_PROP_FPS, 15)
    cap2.set(cv.CAP_PROP_FPS, 15)
    cap1.set(cv.CAP_PROP_AUTOFOCUS, 0)
    cap1.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.75)
    cap2.set(cv.CAP_PROP_AUTOFOCUS, 0)
    cap2.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.75)

    # 启动保存线程池
    save_threads = [Thread(target=save_worker, daemon=True) for _ in range(4)]
    for t in save_threads:
        t.start()

    try:
        for motion_id in range(1, total_motions + 1):
            for trial in range(1, trials_per_motion + 1):
                output_dir = f"data/motion{motion_id}/trial{trial}"
                os.makedirs(output_dir, exist_ok=True)
                
                print(f"开始捕获 motion {motion_id}/{total_motions} trial {trial}/{trials_per_motion}")
                capture_single_sequence(output_dir, kinect, cap1, cap2, disp_width, disp_height)
                
                if not (motion_id == total_motions and trial == trials_per_motion):
                    print(f"等待 {wait_time} 秒...")
                    time.sleep(wait_time)

    finally:
        # 停止保存线程
        for _ in range(4):
            frame_queue.put(None)
        for t in save_threads:
            t.join()
            
        kinect.close()
        cap1.release()
        cap2.release()
        cv.destroyAllWindows()
        hands_detector.close()

if __name__ == '__main__':
    main_automated_capture()
