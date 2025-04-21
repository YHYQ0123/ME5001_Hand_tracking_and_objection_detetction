import cv2 as cv
import glob
import numpy as np
import sys
from scipy import linalg
import os
import pygame
from pykinect2 import PyKinectV2, PyKinectRuntime
import ctypes

# ------------------------- 配置参数 -------------------------
config = {
    "frame_width": 1024,
    "frame_height": 576,
    "mono_calibration_frames": 15,
    "stereo_calibration_frames": 15,
    "view_resize": 1,         # 缩小比例，例如2表示缩小一半
    "cooldown": 50,           # 每次采集帧间隔的冷却计数
    "checkerboard_rows": 7,   # 棋盘格角点行数（不包含外边缘）
    "checkerboard_columns": 10,  # 棋盘格角点列数（不包含外边缘）
    "checkerboard_box_size_scale": 2,  # 棋盘格中每个格子的尺寸缩放因子
    "camera0": 1,             # 第一个 RGB 摄像头设备ID
    "camera1": 2              # 第二个 RGB 摄像头设备ID
}

# ------------------------- 图像变换函数 -------------------------
def transform_frame(frame):
    # 先进行180°翻转（水平和垂直同时翻转）
    cv.rotate(frame, cv.ROTATE_180)
    return frame

# ------------------------- 单摄像头标定帧采集 -------------------------
def save_frames_single_camera(camera_name, config):
    frames_dir = r"C:\Users\xxdbd\Desktop\YeQing\transformer_dataset\frames"
    if not os.path.exists(frames_dir):
        try:
            os.mkdir(frames_dir)
        except Exception as e:
            print(f"Error: Unable to create directory {frames_dir}. {e}")
            return

    camera_device_id = config[camera_name]
    width = config['frame_width']
    height = config['frame_height']
    number_to_save = config['mono_calibration_frames']
    view_resize = config['view_resize']
    cooldown_time = config['cooldown']

    cap = cv.VideoCapture(camera_device_id)
    if not cap.isOpened():
        print(f"Camera {camera_device_id} could not be opened. Skipping...")
        return
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    cooldown = cooldown_time
    start = False
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: No video data received from camera {camera_device_id}. Retrying...")
            continue

        # 如果需要对图像进行翻转，可取消下面这行注释
        frame = transform_frame(frame)

        frame_small = cv.resize(frame, None, fx=1/view_resize, fy=1/view_resize)

        if not start:
            cv.putText(frame_small, "Ensure the checkerboard is fully visible", (50, 50),
                       cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
            cv.putText(frame_small, "Press SPACEBAR to start collecting frames", (50, 100),
                       cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

        if start:
            cooldown -= 1
            cv.putText(frame_small, f"Cooldown: {cooldown}", (50, 50), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
            cv.putText(frame_small, f"Num frames: {saved_count}", (50, 100), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

            if cooldown <= 0:
                savename = os.path.join(frames_dir, f"{camera_name}_{saved_count}.png")
                cv.imwrite(savename, frame)
                saved_count += 1
                cooldown = cooldown_time

        cv.imshow('frame_small', frame_small)
        k = cv.waitKey(1)

        if k == 27:  # ESC 键退出
            print("Exiting...")
            break

        if k == 32:  # SPACEBAR 键开始采集帧
            start = True

        if saved_count == number_to_save:
            break

    cap.release()
    cv.destroyAllWindows()
    print(f"Camera {camera_device_id} frames saved successfully.")

# ------------------------- Kinect 标定帧采集 -------------------------
def save_frames_from_kinect(output_dir, config):
    pygame.init()
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)

    max_frames = config["mono_calibration_frames"]
    cooldown_time = config["cooldown"]
    frame_width = 1920
    frame_height = 1080

    os.makedirs(output_dir, exist_ok=True)

    screen = pygame.display.set_mode((frame_width // 2, frame_height // 2))
    pygame.display.set_caption("Kinect v2 Calibration Frame Capture")

    saved_frames = 0
    cooldown_timer = cooldown_time

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if kinect.has_new_color_frame():
            frame = kinect.get_last_color_frame()
            frame = frame.reshape((frame_height, frame_width, 4))[:, :, :3]  # BGRA 格式

            # 转换为 RGB 格式，适合 Pygame 显示
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            scaled_frame = pygame.transform.scale(frame_surface, (frame_width // 2, frame_height // 2))
            screen.blit(scaled_frame, (0, 0))
            pygame.display.update()

            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE] and cooldown_timer <= 0 and saved_frames < max_frames:
                save_path = os.path.join(output_dir, f"kinect_frame_{saved_frames}.png")
                # 注意保存时转换回 BGR 格式
                cv.imwrite(save_path, cv.cvtColor(frame, cv.COLOR_RGB2BGR))
                saved_frames += 1
                cooldown_timer = cooldown_time
                print(f"Frame {saved_frames} saved: {save_path}")

            cooldown_timer = max(0, cooldown_timer - 1)

            if saved_frames >= max_frames:
                print("All frames saved.")
                break

        clock.tick(30)

    kinect.close()
    pygame.quit()
    print("Kinect closed and program terminated.")

# ------------------------- 内参标定（单摄像头） -------------------------
def calibrate_camera_for_intrinsic_parameters(images_prefix, config):
    images_names = glob.glob(images_prefix)
    print(f"Found images: {images_names}")

    if not images_names:
        print("Error: No images found. Check the images_prefix and ensure files exist.")
        return None, None

    images = [cv.imread(imname, 1) for imname in images_names]
    if not images or any(img is None for img in images):
        print("Error: Failed to load some or all images. Check if the image files are valid.")
        return None, None

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    rows = config['checkerboard_rows']
    columns = config['checkerboard_columns']
    world_scaling = config['checkerboard_box_size_scale']

    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    objpoints = []
    imgpoints = []

    width = images[0].shape[1]
    height = images[0].shape[0]

    for i, frame in enumerate(images):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

        if ret:
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows, columns), corners, ret)
            cv.putText(frame, 'If detected points are poor, press "s" to skip this sample',
                       (25, 25), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            cv.imshow('img', frame)
            k = cv.waitKey(0)
            if k & 0xFF == ord('s'):
                print(f"Skipping image {images_names[i]}...")
                continue

            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print(f"Warning: Chessboard corners not found in image {images_names[i]}. Skipping...")

    cv.destroyAllWindows()

    if not objpoints or not imgpoints:
        print("Error: No valid object points or image points found. Calibration cannot proceed.")
        return None, None

    ret, cmtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print(f"Calibration successful. RMSE: {ret}")
    print("Camera matrix:\n", cmtx)
    print("Distortion coefficients:\n", dist)

    return cmtx, dist

def save_camera_intrinsics(camera_matrix, distortion_coefs, camera_name):
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

    out_filename = os.path.join('camera_parameters', camera_name + '_intrinsics.dat')
    with open(out_filename, 'w') as outf:
        outf.write('intrinsic:\n')
        for l in camera_matrix:
            for en in l:
                outf.write(str(en) + ' ')
            outf.write('\n')
        outf.write('distortion:\n')
        for en in distortion_coefs[0]:
            outf.write(str(en) + ' ')
        outf.write('\n')

# ------------------------- 三摄像头同步标定帧采集 -------------------------
def save_frames_three_cams(camera0_name, camera1_name, camera2_name):
    """
    从 Kinect（camera2_name）、camera0_name、camera1_name 三路读取图像进行同步拍摄。
    不再使用 cooldown，而是按下空格键时才保存当前三路图像各一张。
    按下 ESC 则提前退出拍摄。
    """
    if not os.path.exists('frames_pair'):
        os.mkdir('frames_pair')

    kinect_width, kinect_height = 1920, 1080
    # 显示时缩小的倍率，仅用于可视化，不影响实际保存分辨率
    view_resize = config['view_resize']
    # 希望拍摄多少组标定帧
    number_to_save = config['stereo_calibration_frames']

    # Kinect 读取句柄（只读彩色帧）
    cap0 = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
    # 普通摄像头 0 & 1
    cap1 = cv.VideoCapture(config[camera0_name])
    cap2 = cv.VideoCapture(config[camera1_name])

    width = config['frame_width']
    height = config['frame_height']
    cap1.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap1.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    cap2.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap2.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    saved_count = 0

    while True:
        # 读取 Kinect 帧
        frame0 = cap0.get_last_color_frame()
        # 如果尚未读取到有效帧，frame0 可能是 None
        if frame0 is not None:
            frame0 = frame0.reshape((kinect_height, kinect_width, 4))[:, :, :3]
        else:
            # 若获取不到数据，可根据需求处理（此处仅简单跳过）
            continue

        # 读取两个普通摄像头帧
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            print('Cameras not returning video data. Exiting...')
            break

        # 如果也想对 frame1, frame2 做同样的翻转，可自行取消注释
        frame1 = transform_frame(frame1)
        frame2 = transform_frame(frame2)

        # 仅用于显示：缩小帧
        frame0_small = cv.resize(frame0, None, fx=1./view_resize, fy=1./view_resize)
        frame1_small = cv.resize(frame1, None, fx=1./view_resize, fy=1./view_resize)
        frame2_small = cv.resize(frame2, None, fx=1./view_resize, fy=1./view_resize)

        # 在图像上展示提示信息
        cv.putText(frame0_small, f"Press SPACE to capture, ESC to exit", (50,50),
                   cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
        cv.putText(frame0_small, f"Captured frames: {saved_count}/{number_to_save}", (50,100),
                   cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)

        cv.imshow('Kinect (small)', frame0_small)
        cv.imshow('Camera0 (small)', frame1_small)
        cv.imshow('Camera1 (small)', frame2_small)

        k = cv.waitKey(1) & 0xFF
        if k == 27:  # 按 ESC 退出
            print("User pressed ESC. Exiting capture...")
            break
        elif k == 32:  # 按空格拍照
            # 分别保存三张同步帧
            savename = os.path.join('frames_pair', f"{camera2_name}_{saved_count}.png")  # Kinect
            cv.imwrite(savename, frame0)

            savename = os.path.join('frames_pair', f"{camera0_name}_{saved_count}.png")  # Camera0
            cv.imwrite(savename, frame1)

            savename = os.path.join('frames_pair', f"{camera1_name}_{saved_count}.png")  # Camera1
            cv.imwrite(savename, frame2)

            saved_count += 1
            print(f"[{saved_count}] Synchronized frames saved.")

            if saved_count >= number_to_save:
                print("Reached the desired number of frames. Exiting...")
                break

    cap0.close()
    cap1.release()
    cap2.release()
    cv.destroyAllWindows()


# ------------------------- 新增：挑选有效的三摄像头标定帧 -------------------------
def select_calibration_frames_three_cams(frames_dir, output_dir):
    """
    对采集到的三摄像头同步标定帧进行人工筛选。
    不再拼成一张图，而是分开依次显示每个相机的照片。
    在任何一张图上按 's' 则丢弃该组三张图；
    如果三张图都按空格确认，则将它们保存到 output_dir。
    按 ESC 则直接退出整个挑选流程。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有 Kinect 图像文件（假设命名规则一致，形如 Kinect_0.png、Kinect_1.png 等）
    kinect_images = sorted(glob.glob(os.path.join(frames_dir, "Kinect_*.png")))

    for kinect_img_path in kinect_images:
        basename = os.path.basename(kinect_img_path)   # 如 "Kinect_0.png"
        idx = basename.split('_')[1].split('.')[0]     # 从文件名中获取索引（如 "0"）

        # 找到另外两个相机的对应帧
        cam0_path = os.path.join(frames_dir, f"camera0_{idx}.png")
        cam1_path = os.path.join(frames_dir, f"camera1_{idx}.png")

        # 读取三张图
        img_kinect = cv.imread(kinect_img_path)
        img_cam0   = cv.imread(cam0_path)
        img_cam1   = cv.imread(cam1_path)

        # 如果有缺失就跳过
        if img_kinect is None or img_cam0 is None or img_cam1 is None:
            print(f"Warning: Missing images for index {idx}, skipping...")
            continue

        # 一组三张图的处理逻辑
        skip_set = False  # 是否跳过该组
        images_with_titles = [
            (img_kinect, f"Kinect_{idx}"),
            (img_cam0,   f"camera0_{idx}"),
            (img_cam1,   f"camera1_{idx}")
        ]

        for img, win_name in images_with_titles:
            cv.imshow(win_name, img)
            key = cv.waitKey(0) & 0xFF
            cv.destroyWindow(win_name)   # 关掉当前窗口

            if key == ord('s'):  
                # 用户按 's'，跳过整组
                print(f"Frame set {idx} skipped.")
                skip_set = True
                break
            elif key == 27:
                # 用户按 ESC，退出整个流程
                print("Selection process terminated by user.")
                cv.destroyAllWindows()
                return
            elif key == 32:
                # 空格，用户临时确认该张图，继续查看下一张
                pass
            else:
                # 其他按键也视为跳过
                print(f"Frame set {idx} skipped (unrecognized key).")
                skip_set = True
                break

        # 如果这一组三张图都没被 skip，就保存到输出目录
        if not skip_set:
            # 三张图都按空格确认
            cv.imwrite(os.path.join(output_dir, f"Kinect_{idx}.png"),  img_kinect)
            cv.imwrite(os.path.join(output_dir, f"camera0_{idx}.png"), img_cam0)
            cv.imwrite(os.path.join(output_dir, f"camera1_{idx}.png"), img_cam1)
            print(f"Frame set {idx} accepted.")

    cv.destroyAllWindows()
    print("All sets processed.")


# ------------------------- 立体标定 -------------------------
def stereo_calibrate_three_cameras(kinect_mtx, kinect_dist, left_mtx, left_dist, right_mtx, right_dist, 
                                   frames_prefix_kinect, frames_prefix_left, frames_prefix_right):
    kinect_images_names = sorted(glob.glob(frames_prefix_kinect))
    left_images_names = sorted(glob.glob(frames_prefix_left))
    right_images_names = sorted(glob.glob(frames_prefix_right))

    kinect_images = [cv.imread(imname, 1) for imname in kinect_images_names]
    left_images = [cv.imread(imname, 1) for imname in left_images_names]
    right_images = [cv.imread(imname, 1) for imname in right_images_names]

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    rows = config['checkerboard_rows']
    columns = config['checkerboard_columns']
    world_scaling = config['checkerboard_box_size_scale']

    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    objpoints = []
    imgpoints_kinect = []
    imgpoints_left = []
    imgpoints_right = []

    for kinect_frame, left_frame, right_frame in zip(kinect_images, left_images, right_images):
        gray_kinect = cv.cvtColor(kinect_frame, cv.COLOR_BGR2GRAY)
        gray_left = cv.cvtColor(left_frame, cv.COLOR_BGR2GRAY)
        gray_right = cv.cvtColor(right_frame, cv.COLOR_BGR2GRAY)

        kinect_ret, kinect_corners = cv.findChessboardCorners(gray_kinect, (rows, columns), None)
        left_ret, left_corners = cv.findChessboardCorners(gray_left, (rows, columns), None)
        right_ret, right_corners = cv.findChessboardCorners(gray_right, (rows, columns), None)

        if kinect_ret and left_ret and right_ret:
            kinect_corners = cv.cornerSubPix(gray_kinect, kinect_corners, (11, 11), (-1, -1), criteria)
            left_corners = cv.cornerSubPix(gray_left, left_corners, (11, 11), (-1, -1), criteria)
            right_corners = cv.cornerSubPix(gray_right, right_corners, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints_kinect.append(kinect_corners)
            imgpoints_left.append(left_corners)
            imgpoints_right.append(right_corners)
        else:
            print("Warning: Chessboard corners not found in one set of images, skipping this set.")

    if len(objpoints) == 0:
        print("Error: No valid checkerboard detections in all three cameras.")
        return None, None, None

    ret_kinect_left, kinect_mtx_calibrated, kinect_dist_calibrated, left_mtx_calibrated, left_dist_calibrated, \
        R_kinect_left, T_kinect_left, E_kinect_left, F_kinect_left = cv.stereoCalibrate(
           objpoints, imgpoints_kinect, imgpoints_left, 
           kinect_mtx, kinect_dist, left_mtx, left_dist,
           gray_kinect.shape[::-1], criteria=criteria, flags=cv.CALIB_FIX_INTRINSIC
    )
 
    print("Kinect-Left Stereo Calibration RMS Error:", ret_kinect_left)

    ret_kinect_right, kinect_mtx_calibrated, kinect_dist_calibrated, right_mtx_calibrated, right_dist_calibrated, \
        R_kinect_right, T_kinect_right, E_kinect_right, F_kinect_right = cv.stereoCalibrate(
          objpoints, imgpoints_kinect, imgpoints_right, 
          kinect_mtx, kinect_dist, right_mtx, right_dist,
          gray_kinect.shape[::-1], criteria=criteria, flags=cv.CALIB_FIX_INTRINSIC
    )

    print("Kinect-Right Stereo Calibration RMS Error:", ret_kinect_right)

    cv.destroyAllWindows()

    return R_kinect_left, T_kinect_left, R_kinect_right, T_kinect_right

# ------------------------- 外参保存 -------------------------
def save_extrinsic_calibration_parameters_kinect(R_kinect, T_kinect, R_kinect_left, T_kinect_left, R_kinect_right, T_kinect_right):
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

    kinect_rot_trans_filename = os.path.join('camera_parameters', 'kinect_rot_trans.dat')
    with open(kinect_rot_trans_filename, 'w') as outf:
        outf.write('Kinect R:\n')
        for row in R_kinect:
            outf.write(' '.join(map(str, row)) + '\n')
        outf.write('Kinect T:\n')
        outf.write(' '.join(map(str, T_kinect.flatten())) + '\n')

    kinect_left_filename = os.path.join('camera_parameters', 'kinect_left_rot_trans.dat')
    with open(kinect_left_filename, 'w') as outf:
        outf.write('Kinect to Left Camera R:\n')
        for row in R_kinect_left:
            outf.write(' '.join(map(str, row)) + '\n')
        outf.write('Kinect to Left Camera T:\n')
        outf.write(' '.join(map(str, T_kinect_left.flatten())) + '\n')

    kinect_right_filename = os.path.join('camera_parameters', 'kinect_right_rot_trans.dat')
    with open(kinect_right_filename, 'w') as outf:
        outf.write('Kinect to Right Camera R:\n')
        for row in R_kinect_right:
            outf.write(' '.join(map(str, row)) + '\n')
        outf.write('Kinect to Right Camera T:\n')
        outf.write(' '.join(map(str, T_kinect_right.flatten())) + '\n')

    print("Calibration parameters saved successfully.")

# ------------------------- 主流程 -------------------------
if __name__ == '__main__':
    # Step1.1 Kinect v2 标定帧采集
    output_dir_kinect = r"C:\Users\xxdbd\Desktop\YeQing\transformer_dataset\kinect_calibration_frames"
    save_frames_from_kinect(output_dir_kinect, config)

     # Step1.2 RGB 摄像头标定帧采集（camera0 与 camera1 的设备ID直接在 config 中定义）
    save_frames_single_camera('camera0', config)
    save_frames_single_camera('camera1', config)
    
    # Step2. 计算并保存各摄像头内参
    images_prefix = os.path.join('kinect_calibration_frames', 'kinect_frame*')
    cmtx0, dist0 = calibrate_camera_for_intrinsic_parameters(images_prefix, config)
    save_camera_intrinsics(cmtx0, dist0, 'kinect camera')
    
    images_prefix = os.path.join('frames', 'camera0*')
    cmtx1, dist1 = calibrate_camera_for_intrinsic_parameters(images_prefix, config)
    save_camera_intrinsics(cmtx1, dist1, 'camera0')
    
    images_prefix = os.path.join('frames', 'camera1*')
    cmtx2, dist2 = calibrate_camera_for_intrinsic_parameters(images_prefix, config)
    save_camera_intrinsics(cmtx2, dist2, 'camera1')

    # Step3. 采集三摄像头同步标定帧
    save_frames_three_cams('camera0', 'camera1', 'Kinect')
    
    # 新增步骤：对采集的三摄像头同步标定帧进行挑选（过程类似于 Step2）
    selected_frames_dir = 'selected_frames_pair'
    select_calibration_frames_three_cams('frames_pair', selected_frames_dir)

    # Step4. 利用成对标定帧进行三摄像头立体标定（使用用户挑选后的图片）
    frames_prefix_c0 = os.path.join(selected_frames_dir, 'Kinect*')
    frames_prefix_c1 = os.path.join(selected_frames_dir, 'camera0*')
    frames_prefix_c2 = os.path.join(selected_frames_dir, 'camera1*')
    R_kinect_left, T_kinect_left, R_kinect_right, T_kinect_right = stereo_calibrate_three_cameras(
        cmtx0, dist0, cmtx1, dist1, cmtx2, dist2,
        frames_prefix_c0, frames_prefix_c1, frames_prefix_c2
    )

    # Step5. 以 Kinect 为世界坐标系原点，保存外参
    R_kinect = np.eye(3, dtype=np.float32)
    T_kinect = np.zeros((3, 1), dtype=np.float32)
    save_extrinsic_calibration_parameters_kinect(R_kinect, T_kinect, R_kinect_left, T_kinect_left, R_kinect_right, T_kinect_right)
