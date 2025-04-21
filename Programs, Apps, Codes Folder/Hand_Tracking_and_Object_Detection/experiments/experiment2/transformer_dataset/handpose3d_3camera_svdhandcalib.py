import cv2 as cv
import mediapipe as mp
import numpy as np
from utils3camerasys import get_projection_matrix, write_keypoints_to_disk
from scipy import linalg
from scipy.optimize import least_squares
import os

# Mediapipe工具
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 设定视频帧分辨率（高度×宽度）
frame_shape = [1920, 1080]

# 标准骨骼长度（单位：cm），请根据实际标准模型设置
std_lengths = {
    (3, 4): 2.34, (0, 5): 9.43, (17, 18): 2.86, (0, 17): 8.49,
    (13, 14): 3.81, (13, 17): 1.97, (18, 19): 1.83, (5, 6): 3.84,
    (5, 9): 2.33, (14, 15): 2.30, (0, 1): 4.11, (9, 10): 4.13,
    (1, 2): 3.75, (9, 13): 1.98, (10, 11): 2.53, (19, 20): 1.46,
    (6, 7): 2.20, (15, 16): 1.84, (2, 3): 2.87, (11, 12): 2.04,
    (7, 8): 1.82
}
# 构造所有骨骼对（作为索引使用）
bone_pairs = list(std_lengths.keys())

############################################
# 下面部分实现 DLT+非线性重投影误差优化计算3D点
############################################

def DLT(P1, P2, P3, point1, point2, point3):
    """
    利用三摄像头 DLT 方法求解初始3D点，然后通过非线性优化细化结果。
    参数:
      P1, P2, P3: 三个摄像头的 3×4 投影矩阵（左、右、Kinect）
      point1, point2, point3: 对应摄像头的 2D 图像点 [x, y]
    返回:
      3D 点 [X, Y, Z]
    """
    A = np.zeros((6, 4))
    A[0, :] = point1[1] * P1[2, :] - P1[1, :]
    A[1, :] = P1[0, :] - point1[0] * P1[2, :]
    A[2, :] = point2[1] * P2[2, :] - P2[1, :]
    A[3, :] = P2[0, :] - point2[0] * P2[2, :]
    A[4, :] = point3[1] * P3[2, :] - P3[1, :]
    A[5, :] = P3[0, :] - point3[0] * P3[2, :]

    # 初始解（齐次解归一化）
    U, s, Vh = np.linalg.svd(A)
    X = Vh[-1, :]
    X = X / X[-1]
    initial_3d = X[:3]

    # 通过非线性最小二乘对初始解进行细化
    refined_3d = nonlinear_refinement(P1, P2, P3, point1, point2, point3, initial_3d)
    return refined_3d

def nonlinear_refinement(P1, P2, P3, point1, point2, point3, initial_3d):
    """
    非线性优化细化单个3D点，使得所有视图上的重投影误差最小
    """
    def residual(X):
        res = []
        for P, pt in zip([P1, P2, P3], [point1, point2, point3]):
            X_h = np.hstack((X, 1))
            proj = P @ X_h
            proj = proj / proj[-1]
            res.extend(proj[:2] - pt)
        return res

    result = least_squares(residual, initial_3d)
    return result.x

############################################
# 时序平滑（简单移动平均）
############################################

def temporal_smoothing(keypoints_seq, window_size=5):
    """
    对单个关节在时序数据中的 3D 坐标进行移动平均平滑
    keypoints_seq: (N, 3) 数组
    返回: 平滑后的 (N, 3) 序列
    """
    smoothed = []
    for i in range(len(keypoints_seq)):
        start = max(0, i - window_size + 1)
        window = keypoints_seq[start:i+1]
        smoothed.append(np.mean(window, axis=0))
    return np.array(smoothed)

############################################
# 主函数：多摄像头 2D 检测与 3D 重建
############################################

def run_mp(input_stream_left, input_stream_right, input_stream_kinect, P_left, P_right, P_kinect):
    # 打开三个视频流
    cap_left = cv.VideoCapture(input_stream_left)
    cap_right = cv.VideoCapture(input_stream_right)
    cap_kinect = cv.VideoCapture(input_stream_kinect)
    caps = [cap_left, cap_right, cap_kinect]
    
    # 检查视频是否打开
    if not cap_left.isOpened():
       print("Error: Cannot open left camera video")
       return
    if not cap_right.isOpened():
       print("Error: Cannot open right camera video")
       return
    if not cap_kinect.isOpened():
       print("Error: Cannot open kinect camera video")
       return

    print("🎥 Video files are loaded successfully")

    # 获取帧率、总帧数（可选）
    total_frames_left = int(cap_left.get(cv.CAP_PROP_FRAME_COUNT))
    total_frames_right = int(cap_right.get(cv.CAP_PROP_FRAME_COUNT))
    total_frames_kinect = int(cap_kinect.get(cv.CAP_PROP_FRAME_COUNT))
    fps_left = int(cap_left.get(cv.CAP_PROP_FPS))
    fps_right = int(cap_right.get(cv.CAP_PROP_FPS))
    fps_kinect = int(cap_kinect.get(cv.CAP_PROP_FPS))
    print(f"Left: {total_frames_left} frames, {fps_left} FPS; Right: {total_frames_right} frames, {fps_right} FPS; Kinect: {total_frames_kinect} frames, {fps_kinect} FPS")

    # 创建 Mediapipe 手部检测对象（每个视图单独创建）
    hands_left = mp_hands.Hands(min_detection_confidence=0.6, max_num_hands=1, min_tracking_confidence=0.6)
    hands_right = mp_hands.Hands(min_detection_confidence=0.6, max_num_hands=1, min_tracking_confidence=0.6)
    hands_kinect = mp_hands.Hands(min_detection_confidence=0.6, max_num_hands=1, min_tracking_confidence=0.6)

    # 用于存储各视图的2D关键点和最终计算的3D数据
    kpts_left, kpts_right, kpts_kinect, kpts_3d = [], [], [], []

    while cap_left.isOpened() and cap_right.isOpened() and cap_kinect.isOpened():
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        ret_kinect, frame_kinect = cap_kinect.read()

        if not ret_left or not ret_right or not ret_kinect:
            print("🚨 Error: Video streams not returning frame data. Exiting...")
            break

        # 转换为 RGB（Mediapipe要求RGB输入）
        frame_left_rgb = cv.cvtColor(frame_left, cv.COLOR_BGR2RGB)
        frame_right_rgb = cv.cvtColor(frame_right, cv.COLOR_BGR2RGB)
        frame_kinect_rgb = cv.cvtColor(frame_kinect, cv.COLOR_BGR2RGB)

        # Mediapipe 处理
        results_left = hands_left.process(frame_left_rgb)
        results_right = hands_right.process(frame_right_rgb)
        results_kinect = hands_kinect.process(frame_kinect_rgb)

        # 提取2D关键点函数（返回21个关键点）
        def extract_keypoints(results, frame):
            keypoints = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for p in range(21):
                        pxl_x = int(round(frame.shape[1] * hand_landmarks.landmark[p].x))
                        pxl_y = int(round(frame.shape[0] * hand_landmarks.landmark[p].y))
                        keypoints.append([pxl_x, pxl_y])
            else:
                keypoints = [[-1, -1]] * 21
            return keypoints

        frame_left_keypoints = extract_keypoints(results_left, frame_left)
        frame_right_keypoints = extract_keypoints(results_right, frame_right)
        frame_kinect_keypoints = extract_keypoints(results_kinect, frame_kinect)

        kpts_left.append(frame_left_keypoints)
        kpts_right.append(frame_right_keypoints)
        kpts_kinect.append(frame_kinect_keypoints)

        # 对每个手部关键点，使用三摄像头数据通过 DLT 计算 3D 位置
        frame_p3ds = []
        for uv_left, uv_right, uv_kinect in zip(frame_left_keypoints, frame_right_keypoints, frame_kinect_keypoints):
            if uv_left[0] == -1 or uv_right[0] == -1 or uv_kinect[0] == -1:
                _p3d = [-1, -1, -1]  # 若存在无效点则输出无效值
            else:
                _p3d = DLT(P_left, P_right, P_kinect, uv_left, uv_right, uv_kinect)
            frame_p3ds.append(_p3d)
        # 将当前帧的 3D 数据组织为 (21, 3)
        frame_p3ds = np.array(frame_p3ds).reshape((21, 3))
        kpts_3d.append(frame_p3ds)

        # 绘制手部关键点以便观察（可选）
        def draw_landmarks(frame, results):
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        draw_landmarks(frame_left, results_left)
        draw_landmarks(frame_right, results_right)
        draw_landmarks(frame_kinect, results_kinect)

        # 显示每个摄像头的画面
        cv.imshow('Left Camera', frame_left)
        cv.imshow('Right Camera', frame_right)
        cv.imshow('Kinect Camera', frame_kinect)

        if cv.waitKey(1) & 0xFF == 27:
            break

    cv.destroyAllWindows()
    for cap in caps:
        cap.release()

    # 对 3D 数据序列进行时序平滑处理：对每个关节分别平滑（可选）
    kpts_3d_smoothed = []
    for joint_idx in range(21):
        joint_seq = []
        for frame in kpts_3d:
            if frame[joint_idx][0] != -1:
                joint_seq.append(frame[joint_idx])
        if len(joint_seq) > 0:
            joint_seq = np.array(joint_seq)
            smoothed_joint_seq = temporal_smoothing(joint_seq, window_size=5)
            kpts_3d_smoothed.append(smoothed_joint_seq)
        else:
            kpts_3d_smoothed.append(np.array(joint_seq))
    # 此处 kpts_3d_smoothed 是按关节分开平滑的时序数据，可根据需要重新组织

    # 此处返回原始的 3D 数据序列（未时序平滑）供后续处理
    return np.array(kpts_left), np.array(kpts_right), np.array(kpts_kinect), np.array(kpts_3d)

############################################
# 基于标准骨骼长度进行全局尺度校正
############################################

def calibrate_frame(X, std_lengths, bone_pairs):
    """
    对单帧 Kinect 3D 数据 X（形状 (21, 3)）进行全局尺度校正，
    使得关键骨骼对的长度与标准值相符（单位：cm）。
    返回校正后的 3D 数据 X_calib 和计算得到的尺度因子 s。
    """
    scales = []
    for (i, j) in bone_pairs:
        if np.all(X[i] != -1) and np.all(X[j] != -1):
            measured = np.linalg.norm(X[i] - X[j])
            if measured > 0:
                scales.append(std_lengths[(i, j)] / measured)
    if len(scales) == 0:
        s = 1.0
    else:
        s = np.mean(scales)
    X_calib = X * s
    return X_calib, s

############################################
# 主程序入口
############################################

if __name__ == '__main__':
    # 视频文件路径（请确保路径正确）
    input_stream_left = r'captured_videos\motion_3\cam_1.avi'
    input_stream_right = r'captured_videos\motion_3\cam_2.avi'
    input_stream_kinect = r'captured_videos\motion_3\cam_kinect.avi'

    # 获取各摄像头的投影矩阵（假设 get_projection_matrix 函数已经实现）
    P_left = get_projection_matrix('kinect_left')
    P_right = get_projection_matrix('kinect_right')
    P_kinect = get_projection_matrix('kinect')

    print("✅ All projection matrices are loaded")

    # 运行 Mediapipe 多摄像头处理，得到各视图的 2D 关键点和 3D 数据（未经过全局尺度校正）
    kpts_left, kpts_right, kpts_kinect, kpts_3d = run_mp(input_stream_left, input_stream_right, input_stream_kinect, P_left, P_right, P_kinect)
    print("finish!")

    # 对每一帧 Kinect 得到的 3D 数据进行全局尺度校正
    calibrated_kpts_3d = []
    scale_factors = []
    for frame in kpts_3d:
        frame_calib, s = calibrate_frame(frame, std_lengths, bone_pairs)
        calibrated_kpts_3d.append(frame_calib)
        scale_factors.append(s)
    calibrated_kpts_3d = np.array(calibrated_kpts_3d)

    avg_scale = np.mean(scale_factors) if len(scale_factors) > 0 else 1.0
    print("✅ Calibration done. Average scale factor: {:.3f}".format(avg_scale))

    # 将校正后的 3D 数据写入磁盘（例如保存到文件 kpts_3d_SVD.dat）
    write_keypoints_to_disk('kpts_3d_SVD+handcalib.dat', calibrated_kpts_3d)
