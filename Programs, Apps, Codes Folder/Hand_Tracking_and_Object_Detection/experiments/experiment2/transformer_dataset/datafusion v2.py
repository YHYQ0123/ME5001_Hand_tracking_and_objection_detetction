import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import orthogonal_procrustes

def read_keypoints_kinect(filename):
    """
    读取 Kinect 数据文件 hand_3d.dat。
    文件格式示例（每行）：
      0.071: 0:854.2,697.2,939 1:815.1,675.1,937 2:792.6,640.6,922 ...
    返回形状为 (帧数, 21, 3) 的 numpy 数组（单位：毫米）。
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 忽略行首的时间戳部分（以冒号结尾的部分），提取后面的关键点
            # 使用正则匹配形如 "数字:数字,数字,数字"
            tokens = re.findall(r'\d+:\-?\d+\.?\d*,\-?\d+\.?\d*,\-?\d+\.?\d*', line)
            if len(tokens) != 21:
                continue
            keypoints = []
            for token in tokens:
                parts = token.split(':')
                if len(parts) != 2:
                    continue
                coords_str = parts[1]
                try:
                    coords = list(map(float, coords_str.split(',')))
                except ValueError:
                    continue
                if len(coords) != 3:
                    continue
                keypoints.append(coords)
            if len(keypoints) == 21:
                data.append(np.array(keypoints))
    return np.array(data)

def read_keypoints_3cam(filename):
    """
    读取三摄像头数据文件 kpts_3d.dat。
    假定每行包含 63 个浮点数（21个关键点，每个3个坐标），以空格分隔。
    返回形状为 (帧数, 21, 3) 的 numpy 数组（单位：厘米）。
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                nums = list(map(float, line.split()))
            except ValueError:
                continue
            if len(nums) != 63:
                continue
            pts = np.array(nums).reshape((21, 3))
            data.append(pts)
    return np.array(data)

def rigid_transform(A, B):
    """
    对应点集 A 与 B（形状：(n, 3)）求刚性变换（旋转 R 与平移 t），使得
      B ≈ A * R + t
    使用 scipy.linalg.orthogonal_procrustes 实现。
    返回 R (3x3) 和 t (1x3)。
    """
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    A_centered = A - centroid_A
    B_centered = B - centroid_B
    R, _ = orthogonal_procrustes(A_centered, B_centered)
    t = centroid_B - centroid_A.dot(R)
    return R, t

def fuse_keypoints(kpts_3cam, kpts_kinect, weight_3cam=0.5, weight_kinect=0.5):
    """
    融合两组 3D 手部关键点数据：
      - kpts_3cam：三摄像头数据，单位：厘米，形状为 (N, 21, 3)
      - kpts_kinect：Kinect 数据，单位：毫米，形状为 (N, 21, 3)
    1. 将 Kinect 数据转换为厘米；
    2. 对每一帧，通过刚性配准将 Kinect 数据对齐到三摄像头数据的坐标系；
    3. 按权重进行加权平均融合。
    返回融合后的数据，形状为 (N, 21, 3)。
    """
    # Kinect 单位转换：毫米 -> 厘米
    kpts_kinect_cm = kpts_kinect / 10.0

    num_frames = min(kpts_3cam.shape[0], kpts_kinect_cm.shape[0])
    fused = []

    for i in range(num_frames):
        X_ref = kpts_3cam[i]           # 三摄像头数据（参考），单位：厘米
        X_kinect = kpts_kinect_cm[i]   # Kinect 数据（单位：厘米）
        # 刚性配准：求得旋转 R 和平移 t，使得 X_kinect 对齐到 X_ref
        R, t = rigid_transform(X_kinect, X_ref)
        X_kinect_aligned = X_kinect.dot(R) + t
        # 按权重加权融合
        fused_frame = weight_3cam * X_ref + weight_kinect * X_kinect_aligned
        fused.append(fused_frame)

    return np.array(fused)

def plot_hand(ax, kpts, color='green', marker='o'):
    """
    在给定的 3D 坐标系 ax 中绘制手部关键点及骨架。
    这里假定关键点顺序为：0 为手掌中心，
    1-4：拇指，5-8：食指，9-12：中指，13-16：无名指，17-20：小指。
    """
    fingers = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20]
    ]
    for finger in fingers:
        pts = kpts[finger]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, marker=marker)
    ax.scatter(kpts[:, 0], kpts[:, 1], kpts[:, 2], color=color)

if __name__ == '__main__':
    # 读取数据文件（确保文件路径正确）
    kpts_3cam = read_keypoints_3cam('kpts_3d.dat')   # 三摄像头数据（单位：厘米）
    kpts_kinect = read_keypoints_kinect('hand_3d.dat')  # Kinect 数据（单位：毫米）

    # 保证两组数据帧数一致
    num_frames = min(kpts_3cam.shape[0], kpts_kinect.shape[0])
    kpts_3cam = kpts_3cam[:num_frames]
    kpts_kinect = kpts_kinect[:num_frames]

    # 数据融合：返回融合后数据，单位：厘米
    fused_kpts = fuse_keypoints(kpts_3cam, kpts_kinect, weight_3cam=0.5, weight_kinect=0.5)

    # 计算所有帧融合数据的全局坐标范围，用于统一坐标轴
    all_points = fused_kpts.reshape(-1, 3)
    xmin, ymin, zmin = all_points.min(axis=0)
    xmax, ymax, zmax = all_points.max(axis=0)

    # 创建 3D 图形
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()
        plot_hand(ax, fused_kpts[frame], color='green')
        ax.set_title(f'Fused Data - Frame {frame}')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Z (cm)')
        # 可根据需要设置视角
        ax.view_init(elev=90, azim=90)

    # 使用 FuncAnimation 对所有帧进行动态显示
    ani = FuncAnimation(fig, update, frames=num_frames, interval=100, repeat=True)
    plt.show()
