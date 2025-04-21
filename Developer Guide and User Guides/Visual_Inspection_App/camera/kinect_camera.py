# -*- coding: utf-8 -*-
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
import numpy as np
import time
import sys
import mediapipe as mp
import cv2 # 明确导入 cv2


# --- 尝试导入 Kinect V2 库 ---
try:
    from pykinect2 import PyKinectV2
    from pykinect2.PyKinectV2 import *
    from pykinect2 import PyKinectRuntime
    PYKINECT2_AVAILABLE = True
    print("PyKinectV2 库导入成功。")
except ImportError:
    print("警告: 未找到 PyKinectV2 库。Kinect 将运行在模拟模式。", file=sys.stderr)
    PYKINECT2_AVAILABLE = False

# --- Mediapipe 初始化 ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# --- 定义相机线程类 ---
class KinectCameraThread(QThread):
    PYKINECT2_AVAILABLE = True
    frame_signal = pyqtSignal(object)      # 发射处理后的 RGB numpy 数组帧
    status_signal = pyqtSignal(str)        # (可选) 发射状态文本信号
    recording_state_signal = pyqtSignal(bool) # (可选) 发射录制状态信号
    error_signal = pyqtSignal(str, str)    # 发射 camera_id, 错误信息
    finished_signal = pyqtSignal()          # 发射线程结束信号
    
    request_app_stop = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera_id = "KinectV2"
        self._is_running = False
        self.kinect_runtime = None # Kinect 运行时实例
        print(f"初始化 {self.camera_id}...")

        # --- 状态、模型和处理控制变量 ---
        self.current_stage_index = 0
        self.status_text = "Waiting to start..." # 等待开始
        self.should_record = False
        self.detected_regions_flags = [False] * 5
        self.frame_count = 0               # 帧计数器
        self.processing_interval = 2       # 每隔几帧处理一次
        # 用于保存上一次的处理结果，以便在非处理帧上也能显示
        self.last_best_box_orig = None
        self.last_best_label = ""
        self.last_hands_data = [] # 存储上次检测到的手部数据（包含左右手信息和原始坐标点）

        # --- *** 新增：用于存储手部关键点数据的列表 *** ---
        # 包含 5 个列表，分别对应 5 个录制阶段
        self.recorded_hand_data = [[] for _ in range(5)]
        # Example structure:
        # self.recorded_hand_data[0] = [ # Stage 1 data
        #   [(x0,y0), (x1,y1), ... (x20,y20)], # Frame 1 landmarks for right hand
        #   [(x0,y0), (x1,y1), ... (x20,y20)], # Frame 2 landmarks for right hand
        #   ...
        # ]
        # self.recorded_hand_data[1] = [ ... ] # Stage 2 data
        # ... and so on for stages 3, 4, 5.

        # --- 感兴趣区域 (ROI) 定义 (基于调整后的 660x540 帧) ---
        self.regions_on_resized = [
            [(450, 376), (530, 427)],  # 区域1
            [(330, 384), (410, 425)],  # 区域2
            [(210, 377), (296, 424)],  # 区域3
            [(85, 378), (176, 423)],   # 区域4
            [(83, 275), (175, 329)],   # 区域5
        ]
        self.region_colors = [(255, 255, 0), (255, 255, 0), (255, 255, 0), (255, 255, 0), (255, 255, 0)] # BGR, Cyan
        self.highlight_region_number = 0 # 当前需要高亮的目标区域索引
        self.highlight_region_color = (0, 0, 255) # BGR, Red for highlight

        # --- 加载模型 ---
        self.model_group = []
        self.model_single = None
        try:
            print("正在加载 YOLO 模型...")
            models_to_load = [
                r'models\0_1.pt', r'models\1_2.pt', r'models\2_3.pt',
                r'models\3_4.pt', r'models\4_5.pt'
            ]
            for model_path in models_to_load:
                self.model_group.append(YOLO(model_path).to('cuda'))
            self.model_single = YOLO(r'models\single.pt').to('cuda')
            print("YOLO 模型加载成功。")
        except Exception as e:
            print(f"错误: 加载 YOLO 模型失败: {e}", file=sys.stderr)
            self.error_signal.emit(self.camera_id, f"模型加载失败: {e}")
            self.model_group = []
            self.model_single = None

        # --- 初始化 Mediapipe Hands ---
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)

        if not PYKINECT2_AVAILABLE:
            print("警告: Kinect 运行在模拟模式。", file=sys.stderr)

    # --- 坐标转换辅助函数 ---
    def map_resized_to_original(self, cx_resized, cy_resized, crop_x_offset=440, scale_factor=2):
        original_x = int(cx_resized * scale_factor + crop_x_offset)
        original_y = int(cy_resized * scale_factor)
        return original_x, original_y

    def run(self):
        self._is_running = True
        print(f"启动摄像头线程: {self.camera_id}")
        try:
            if PYKINECT2_AVAILABLE:
                self.kinect_runtime = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
                if self.kinect_runtime is None:
                        raise RuntimeError("无法初始化 Kinect V2 运行时。")
            elif not self.model_group:
                    raise RuntimeError("模拟模式需要成功加载的模型。")

            # --- 主循环 ---
            while self._is_running:

                frame_bgr = None
                frame_resized = None

                # --- 1. 获取帧 ---
                if PYKINECT2_AVAILABLE and self.kinect_runtime and self.kinect_runtime.has_new_color_frame():
                    frame_1d = self.kinect_runtime.get_last_color_frame()
                    frame_bgra = frame_1d.reshape((self.kinect_runtime.color_frame_desc.Height,
                                                   self.kinect_runtime.color_frame_desc.Width, 4))
                    frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
                elif not PYKINECT2_AVAILABLE:
                    frame_bgr = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
                    cv2.putText(frame_bgr, "Kinect SIM Mode", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else:
                    QThread.msleep(10)
                    continue

                # --- 帧获取成功后增加计数 ---
                self.frame_count += 1

                # --- 2. 预处理 (裁剪和缩放) - 每帧都做 ---
                if frame_bgr is None: continue
                crop_y_start, crop_y_end = 0, 1080
                crop_x_start, crop_x_end = 440, 1760
                resize_w, resize_h = 660, 540
                frame_cropped = frame_bgr[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
                frame_resized = cv2.resize(frame_cropped, (resize_w, resize_h))

                # --- 准备输出帧 (用于绘制) ---
                output_frame = frame_bgr.copy()

                # --- 3. 绘制 ROI 区域 (每帧都绘制, 高亮当前目标区域) ---
                for idx, (pt1_resized, pt2_resized) in enumerate(self.regions_on_resized):
                    orig_x1, orig_y1 = self.map_resized_to_original(pt1_resized[0], pt1_resized[1], crop_x_start, scale_factor=2)
                    orig_x2, orig_y2 = self.map_resized_to_original(pt2_resized[0], pt2_resized[1], crop_x_start, scale_factor=2)

                    color = self.highlight_region_color if idx == self.highlight_region_number else self.region_colors[idx]
                    thickness = 5 if idx == self.highlight_region_number else 3

                    cv2.rectangle(output_frame, (orig_x1, orig_y1), (orig_x2, orig_y2), color, thickness)
                    cv2.putText(output_frame, f"Area_{idx+1}", (orig_x1 + 5, orig_y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # --- 4. 条件处理块 (每 N 帧执行一次) ---
                if self.frame_count % self.processing_interval == 0:

                    # --- 4.1 YOLO 推理 ---
                    best_class, best_conf, best_box_resized = None, 0.0, None
                    current_model = self.model_group[self.current_stage_index] if self.current_stage_index < len(self.model_group) else None
                    if current_model:
                        results = current_model(frame_resized, verbose=False)
                        for result in results:
                            for box in result.boxes:
                                conf, cls_idx = float(box.conf[0]), int(box.cls[0])
                                if conf > best_conf and conf > 0.90: # 置信度阈值 0.9
                                    best_conf = conf
                                    best_class = current_model.names[cls_idx]
                                    best_box_resized = box.xyxy[0].cpu().numpy()

                    # 更新用于绘制的持久化变量
                    if best_box_resized is not None:
                         x1_res, y1_res, x2_res, y2_res = map(int, best_box_resized)
                         orig_bx1, orig_by1 = self.map_resized_to_original(x1_res, y1_res, crop_x_start, scale_factor=2)
                         orig_bx2, orig_by2 = self.map_resized_to_original(x2_res, y2_res, crop_x_start, scale_factor=2)
                         self.last_best_box_orig = (orig_bx1, orig_by1, orig_bx2, orig_by2)
                         self.last_best_label = f"{best_class} {best_conf:.2f}"
                    else:
                         self.last_best_box_orig = None
                         self.last_best_label = ""

                    # --- 4.2 Mediapipe 手部检测 ---
                    rgb_frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    rgb_frame_resized.flags.writeable = False
                    results_hands = self.hands.process(rgb_frame_resized)
                    rgb_frame_resized.flags.writeable = True

                    self.detected_regions_flags = [False] * len(self.regions_on_resized)
                    current_hands_data = [] # 存储当前帧的手部数据用于绘制
                    # checked_region = False # 用于检查手是否已经在区域内 (旧逻辑，现在直接使用detected_regions_flags)

                    if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
                        for hand_idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                            handedness = results_hands.multi_handedness[hand_idx]
                            hand_label = handedness.classification[0].label
                            hand_score = handedness.classification[0].score

                            landmarks_for_this_hand_orig = []
                            current_frame_right_hand_coords = [] # 临时列表，用于存储当前帧的右手坐标

                            for idx, landmark in enumerate(hand_landmarks.landmark):
                                cx_res, cy_res = int(landmark.x * resize_w), int(landmark.y * resize_h)
                                orig_lx, orig_ly = self.map_resized_to_original(cx_res, cy_res, crop_x_start, scale_factor=2)

                                # --- 存储映射回原始坐标的点用于绘制 (所有手) ---
                                landmarks_for_this_hand_orig.append((orig_lx, orig_ly))

                                # --- *** 修改点：区域检查和数据记录 *** ---
                                if hand_label == "Right":
                                    # 检查右手是否在当前目标区域内
                                    target_region_idx = self.current_stage_index
                                    if target_region_idx < len(self.regions_on_resized):
                                        r_x1, r_y1 = self.regions_on_resized[target_region_idx][0]
                                        r_x2, r_y2 = self.regions_on_resized[target_region_idx][1]
                                        # 检查当前 landmark 是否在调整后的目标区域内
                                        if r_x1 < cx_res < r_x2 and r_y1 < cy_res < r_y2:
                                            # 只要右手的任何一个关键点在目标区域内，就认为检测到
                                            self.detected_regions_flags[target_region_idx] = True
                                            # 注意：我们不在检测到第一个点后就break，因为我们需要收集*所有*右手关键点的数据

                                    # --- *** 新增：如果处于录制状态，则记录右手坐标 *** ---
                                    if self.should_record:
                                        # 将当前右手关键点的 *原始坐标* 添加到临时列表
                                        current_frame_right_hand_coords.append((orig_lx, orig_ly))


                            # --- *** 新增：完成一只手的处理后 *** ---
                            # 如果是右手 且 处于录制状态 且 收集到了坐标点
                            if hand_label == "Right" and self.should_record and current_frame_right_hand_coords:
                                # 检查当前阶段索引是否有效
                                if 0 <= self.current_stage_index < len(self.recorded_hand_data):
                                     # 将当前帧收集到的所有右手关键点坐标追加到对应阶段的列表中
                                    self.recorded_hand_data[self.current_stage_index].append(current_frame_right_hand_coords)
                                    # print(f"DEBUG: Recorded frame for stage {self.current_stage_index+1}, total frames: {len(self.recorded_hand_data[self.current_stage_index])}") # 可选的调试输出
                                else:
                                    print(f"警告: 无效的 current_stage_index ({self.current_stage_index}) 无法存储手部数据。", file=sys.stderr)


                            # 将当前手的数据（左右手标签，原始坐标点）添加到 *用于绘制* 的列表
                            current_hands_data.append({
                                "label": hand_label,
                                "score": hand_score,
                                "landmarks_orig": landmarks_for_this_hand_orig
                            })

                    # 更新持久化的手部数据，用于非处理帧的绘制
                    self.last_hands_data = current_hands_data

                    # --- 4.3 更新状态逻辑 ---
                    prev_status = self.status_text
                    prev_record_state = self.should_record
                    # 检查当前目标区域是否被右手触发
                    current_target_region_detected = False
                    if 0 <= self.current_stage_index < len(self.detected_regions_flags):
                        current_target_region_detected = self.detected_regions_flags[self.current_stage_index]

                    # --- *** 状态更新逻辑 (根据 YOLO 结果和右手区域检测更新 should_record) *** ---
                    next_should_record = False # 默认为 False
                    next_status_text = self.status_text # 默认保持不变

                    if best_class == 'stage_1':
                        self.highlight_region_number = 0
                        if current_target_region_detected:
                            next_status_text = "Recording gesture 1..."
                            next_should_record = True
                        else:
                            next_status_text = "Please move RIGHT hand to Area 1"
                            # next_should_record 保持 False
                    elif best_class == 'stage_2':
                        self.highlight_region_number = 1
                        if current_target_region_detected:
                            next_status_text = "Recording gesture 2..."
                            next_should_record = True
                        else:
                            # 检查是否可以从上一阶段推进
                            if self.current_stage_index == 0:
                                print("Stage advancing from 1 to 2")
                                next_status_text = "Gesture 1 recorded. Move RIGHT hand to Area 2"
                                self.current_stage_index = 1 # 更新阶段
                                # next_should_record 保持 False
                            else: # 如果还在当前阶段，但手不在区域内
                                next_status_text = "Please move RIGHT hand to Area 2"
                                # next_should_record 保持 False
                    elif best_class == 'stage_3':
                        self.highlight_region_number = 2
                        if current_target_region_detected:
                            next_status_text = "Recording gesture 3..."
                            next_should_record = True
                        else:
                            if self.current_stage_index == 1:
                                print("Stage advancing from 2 to 3")
                                next_status_text = "Gesture 2 recorded. Move RIGHT hand to Area 3"
                                self.current_stage_index = 2
                            else:
                                next_status_text = "Please move RIGHT hand to Area 3"
                    elif best_class == 'stage_4':
                        self.highlight_region_number = 3
                        if current_target_region_detected:
                            next_status_text = "Recording gesture 4..."
                            next_should_record = True
                        else:
                            if self.current_stage_index == 2:
                                print("Stage advancing from 3 to 4")
                                next_status_text = "Gesture 3 recorded. Move RIGHT hand to Area 4"
                                self.current_stage_index = 3
                            else:
                                next_status_text = "Please move RIGHT hand to Area 4"
                    elif best_class == 'stage_5':
                        self.highlight_region_number = 4
                        if current_target_region_detected:
                            next_status_text = "Recording gesture 5..."
                            next_should_record = True
                        else:
                            if self.current_stage_index == 3:
                                print("Stage advancing from 4 to 5")
                                next_status_text = "Gesture 4 recorded. Move RIGHT hand to Area 5"
                                self.current_stage_index = 4
                            else:
                                next_status_text = "Please move RIGHT hand to Area 5"
                    elif best_class == 'stage_6':
                        self.highlight_region_number = -1 # No highlight
                        if self.current_stage_index == 4:
                            print("Stage advancing from 5 to completion")
                            next_status_text = "All gestures recorded. Complete!"
                            for i, data_list in enumerate(self.recorded_hand_data):
                                print(f"Stage {i+1}: Recorded {len(data_list)} frames.")
                            self.request_app_stop.emit()
                            self.stop()
                        else:
                            next_status_text = "Your assembly is false in stage"+str(self.current_stage_index+1)
                        next_should_record = False # Stop recording upon completion
                    else: # No specific stage detected or invalid state
                        self.highlight_region_number = self.current_stage_index # Highlight current target
                        # Maintain current status unless hand is in the correct zone,
                        # but don't necessarily start recording unless YOLO agrees.
                        # Keep should_record False if YOLO doesn't match the expected next stage.
                        next_should_record = False # Default to not recording if YOLO doesn't match
                        # Update status text based on current stage index
                        if 0 <= self.current_stage_index < 5:
                            next_status_text = f"Please perform gesture for Stage {self.current_stage_index + 1} and move RIGHT hand to Area {self.current_stage_index + 1}"
                        else: # Should not happen if logic is correct
                             next_status_text = "Waiting..."


                    # --- *** 应用状态和录制标志更新 *** ---
                    self.status_text = next_status_text
                    self.should_record = next_should_record


                    # 发射状态变化信号
                    if self.status_text != prev_status:
                        self.status_signal.emit(self.status_text)
                    if self.should_record != prev_record_state:
                        self.recording_state_signal.emit(self.should_record)
                        # --- 调试信息 ---
                        if self.should_record:
                             print(f"INFO: Started recording for stage {self.current_stage_index + 1}")
                        elif prev_record_state: # Was recording, now stopped
                             print(f"INFO: Stopped recording for stage {self.current_stage_index + 1}") # Note: index might have just changed

                # --- 5. 绘制动态/持久化元素 (每帧都绘制) ---
                # 绘制 YOLO 框
                if self.last_best_box_orig is not None:
                    orig_bx1, orig_by1, orig_bx2, orig_by2 = self.last_best_box_orig
                    cv2.rectangle(output_frame, (orig_bx1, orig_by1), (orig_bx2, orig_by2), (255, 0, 0), 3)
                    cv2.putText(output_frame, self.last_best_label, (orig_bx1, orig_by1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2) # Yellow label

                # 绘制手 (使用上一次处理的结果 self.last_hands_data)
                if self.last_hands_data:
                    connections = self.mp_hands.HAND_CONNECTIONS
                    for hand_data in self.last_hands_data:
                        hand_label = hand_data["label"]
                        landmarks_orig = hand_data["landmarks_orig"]
                        hand_color = (0, 0, 255) if hand_label == "Right" else (255, 255, 0) # Red for Right, Cyan for Left
                        connection_color = (0, 0, 255) if hand_label == "Right" else (255, 255, 0)

                        wrist_pos = None
                        for idx, (orig_lx, orig_ly) in enumerate(landmarks_orig):
                            cv2.circle(output_frame, (orig_lx, orig_ly), 5, hand_color, -1)
                            if idx == 0: wrist_pos = (orig_lx, orig_ly)

                        for connection in connections:
                            start_idx, end_idx = connection[0], connection[1]
                            if start_idx < len(landmarks_orig) and end_idx < len(landmarks_orig):
                                start_pt, end_pt = landmarks_orig[start_idx], landmarks_orig[end_idx]
                                cv2.line(output_frame, start_pt, end_pt, connection_color, 2)

                        if wrist_pos:
                            cv2.putText(output_frame, hand_label, (wrist_pos[0] - 40, wrist_pos[1] + 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, hand_color, 2)

                # --- 6. 绘制当前状态文本 (每帧都绘制) ---
                ts = cv2.getTextSize(self.status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                cv2.rectangle(output_frame, (30, 40-ts[1]), (30+ts[0]+20, 60), (255,255,255), -1)
                cv2.putText(output_frame, self.status_text, (35, 55-ts[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 3)

                # --- 7. 发射帧信号 (每帧都发射) ---
                frame_rgb_out = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                self.frame_signal.emit(frame_rgb_out)

        except Exception as e:
            error_msg = f"摄像头 {self.camera_id} 运行时错误: {e}"
            import traceback
            print(error_msg, file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            self.error_signal.emit(self.camera_id, str(e))
        finally:
                # --- 清理 ---
                print(f"Entering finally block for {self.camera_id}...") # Add debug print
                if PYKINECT2_AVAILABLE and self.kinect_runtime:
                    try: # Add try-except for closing runtime
                        self.kinect_runtime.close()
                        print(f"Kinect V2 运行时已关闭: {self.camera_id}")
                    except Exception as close_e:
                        print(f"Error closing Kinect runtime: {close_e}", file=sys.stderr)

                if hasattr(self, 'hands') and self.hands:
                    try: # Add try-except for closing hands
                        self.hands.close()
                        print("Mediapipe Hands instance closed.")
                    except Exception as hands_e:
                        print(f"Error closing Mediapipe hands: {hands_e}", file=sys.stderr)

                # 确保 _is_running 为 False
                self._is_running = False
                # 发射结束信号
                print(f"Emitting finished signal for {self.camera_id}...") # Add debug print
                self.finished_signal.emit()
                print(f"摄像头线程已结束: {self.camera_id}")


    def stop(self):
        print(f"请求停止摄像头线程: {self.camera_id}")
        self._is_running = False

    # --- *** 新增: 方法来获取存储的数据 *** ---
    def get_recorded_data(self):
        """返回存储的所有阶段的手部关键点数据"""
        return self.recorded_hand_data

    def clear_recorded_data(self):
        """清空所有已记录的数据"""
        print("Clearing recorded hand data.")
        self.recorded_hand_data = [[] for _ in range(5)]