import numpy as np
from scipy import linalg

def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
    return P

# Direct linear transform
def DLT(P1, P2, point1, point2):
    
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))
    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices=False)
    
    return Vh[-1, :3] / Vh[-1, 3]

def read_camera_parameters(camera_name):
    # 构造文件路径
    file_path = f'camera_parameters/{camera_name}_intrinsics.dat'
    with open(file_path, 'r') as inf:
        cmtx = []  # 存储内参矩阵
        dist = []  # 存储畸变系数

        # 跳过非数值行（例如 'intrinsic:' 和 'distortion:'）
        for line in inf:
            line = line.strip()  # 去掉首尾空格
            if line == "intrinsic:":  # 内参部分开始
                # 读取 3 行内参矩阵
                for _ in range(3):
                    cmtx.append([float(num) for num in inf.readline().strip().split()])
            elif line == "distortion:":  # 畸变系数部分开始
                # 读取 1 行畸变系数
                dist.append([float(num) for num in inf.readline().strip().split()])
    
    return np.array(cmtx), np.array(dist)


def read_rotation_translation(camera_name, savefolder='camera_parameters/'):
    file_path = f"{savefolder}{camera_name}_rot_trans.dat"
    with open(file_path, 'r') as inf:
        lines = inf.readlines()

    rot = []
    trans = []

    for i, line in enumerate(lines):
        if 'R:' in line:
            # 读取旋转矩阵的三行
            rot = [
                [float(x) for x in lines[i + 1].split()],
                [float(x) for x in lines[i + 2].split()],
                [float(x) for x in lines[i + 3].split()]
            ]
        if 'T:' in line:
            # 读取平移向量的一行
            trans = [float(x) for x in lines[i + 1].split()]

    if not rot or not trans:
        raise ValueError(f"Invalid or incomplete data in {file_path}. Ensure the file contains valid 'R:' and 'T:' sections.")

    
    return np.array(rot), np.array(trans).reshape(3, 1)





def _convert_to_homogeneous(pts):
    pts = np.array(pts)
    if len(pts.shape) > 1:
        w = np.ones((pts.shape[0], 1))
        return np.concatenate([pts, w], axis=1)
    else:
        return np.concatenate([pts, [1]], axis=0)

def get_projection_matrix(camera_name):
    """计算相机的投影矩阵"""
    
    cmtx, dist = read_camera_parameters(camera_name)
    R, T = read_rotation_translation(camera_name)
    P = cmtx @ _make_homogeneous_rep_matrix(R, T)[:3, :]
    return P

def write_keypoints_to_disk(filename, kpts):
    """将关键点保存到磁盘"""
    with open(filename, 'w') as fout:
        for frame_kpts in kpts:
            for kpt in frame_kpts:
                if len(kpt) == 2:
                    fout.write(f"{kpt[0]} {kpt[1]} ")
                else:
                    fout.write(f"{kpt[0]} {kpt[1]} {kpt[2]} ")
            fout.write('\n')

if __name__ == '__main__':
    # Kinect 相机
    P_kinect = get_projection_matrix('kinect')
    # 左相机
    P_left = get_projection_matrix('kinect_left')
    # 右相机
    P_right = get_projection_matrix('kinect_right')

    print("Kinect Projection Matrix:", P_kinect)
    print("Left Camera Projection Matrix:", P_left)
    print("Right Camera Projection Matrix:", P_right)

   
