# -*- coding: utf-8 -*-
from PyQt5.QtCore import QThread, pyqtSignal
import time
import sys
import os # Import os for path joining

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("Error: OpenCV (cv2) library not found. Cannot use OpenCV cameras.", file=sys.stderr)
    CV2_AVAILABLE = False

import config  # 导入分辨率设置的配置

class OpenCVCameraThread(QThread):
    frame_signal = pyqtSignal(object)  # 发射 RGB numpy 数组
    error_signal = pyqtSignal(str, str)  # (相机ID, 错误信息)
    finished_signal = pyqtSignal()  # 标志线程完成

    # --- NEW: Signal to report recording status ---
    recording_status_signal = pyqtSignal(str, bool) # camera_id, is_recording

    def __init__(self, camera_index, parent=None):
        super().__init__(parent)
        self.source = camera_index
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV is required for OpenCVCameraThread.")

        self.camera_index = camera_index
        self.camera_id = f"OpenCV_{camera_index}"
        self._is_running = False
        self.cap = None

        # --- NEW: Recording state variables ---
        self.is_recording = False
        self.video_writer = None
        self.output_filename = None
        self.frame_width = config.FRAME_WIDTH # Store dimensions for writer
        self.frame_height = config.FRAME_HEIGHT
        self.fps = 30.0 # Target FPS for recording (adjust as needed)
        # --- End NEW ---

        print(f"Initializing {self.camera_id}...")

    # --- NEW: Method to start recording ---
    def start_recording(self, filename):
        if not CV2_AVAILABLE:
            print(f"Error: Cannot start recording for {self.camera_id}, OpenCV not available.", file=sys.stderr)
            return False
        if not self.cap or not self.cap.isOpened():
            print(f"Error: Cannot start recording for {self.camera_id}, camera not open.", file=sys.stderr)
            return False
        if self.is_recording:
            print(f"Warning: {self.camera_id} is already recording.", file=sys.stderr)
            return True # Already recording

        self.output_filename = filename
        # Use actual dimensions obtained after opening the camera
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Try getting actual FPS, fallback to default
        cap_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if cap_fps > 0:
            self.fps = cap_fps
            print(f"{self.camera_id} using actual FPS for recording: {self.fps}")
        else:
            print(f"{self.camera_id} using default FPS for recording: {self.fps}")


        # Define the codec and create VideoWriter object
        # Common codecs: 'XVID', 'MJPG', 'MP4V' (for .mp4), 'DIVX'
        fourcc = cv2.VideoWriter_fourcc(*'XVID') # Using XVID for .avi
        try:
            # Crucially, VideoWriter expects BGR frames
            self.video_writer = cv2.VideoWriter(self.output_filename, fourcc, self.fps, (width, height))
            if not self.video_writer.isOpened():
                 raise IOError(f"Could not open video writer for {self.output_filename}")
            self.is_recording = True
            print(f"{self.camera_id} started recording to {self.output_filename}")
            self.recording_status_signal.emit(self.camera_id, True)
            return True
        except Exception as e:
            print(f"Error creating VideoWriter for {self.camera_id}: {e}", file=sys.stderr)
            self.video_writer = None
            self.output_filename = None
            self.is_recording = False
            self.error_signal.emit(self.camera_id, f"Failed to start recording: {e}")
            return False

    # --- NEW: Method to stop recording ---
    def stop_recording(self):
        if not self.is_recording or self.video_writer is None:
            # print(f"Debug: Stop recording called for {self.camera_id}, but not recording.")
            return

        print(f"{self.camera_id} stopping recording...")
        self.is_recording = False # Set flag first
        time.sleep(0.1) # Allow potential last frame write
        try:
            if self.video_writer and self.video_writer.isOpened():
                 self.video_writer.release()
                 print(f"{self.camera_id} finished recording to {self.output_filename}")
            else:
                 print(f"Warning: VideoWriter for {self.camera_id} was not open or None on stop.")
        except Exception as e:
            print(f"Error releasing video writer for {self.camera_id}: {e}", file=sys.stderr)
        finally:
             self.video_writer = None
             saved_filename = self.output_filename
             self.output_filename = None
             self.recording_status_signal.emit(self.camera_id, False)
             # Optionally return the filename: return saved_filename


    def run(self):
        if self.camera_index =="video\cam1.avi" or self.camera_index == "video\cam2.avi"or self.camera_index == "video\kinect.avi":
            self.show()
        else:
            self._is_running = True
            print(f"Starting camera thread: {self.camera_id}")
            try:
                # Attempt using DSHOW first (often better performance/control on Windows)
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    print(f"Warning: Failed to open {self.camera_id} with DSHOW, trying default backend...")
                    self.cap = cv2.VideoCapture(self.camera_index)

                if not self.cap.isOpened():
                    raise IOError(f"Cannot open camera {self.camera_index}")

                # Set desired resolution from config
                set_w = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
                set_h = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
                # Set FPS (might not be respected by all cameras/backends)
                set_fps = self.cap.set(cv2.CAP_PROP_FPS, self.fps) # Use the class fps attribute

                # Verify actual settings
                self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS) # Get actual FPS after setting
                if not set_w or not set_h:
                    print(f"Warning: {self.camera_id} may not support setting resolution to {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}.")
                if not set_fps:
                    print(f"Warning: {self.camera_id} may not support setting FPS to {self.fps}.")

                print(f"{self.camera_id} running at {self.frame_width}x{self.frame_height} @ {actual_fps:.2f} FPS (requested {self.fps:.2f})")
                # Update self.fps if camera reported a different value? Optional.
                # if actual_fps > 0: self.fps = actual_fps

                while self._is_running:
                    ret, frame_bgr = self.cap.read()
                    if ret:
                        # --- Perform operations needed for display (like rotation) ---
                        # Convert BGR to RGB for Qt display and potential processing
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        # Rotate *after* potentially writing the original BGR frame
                        frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_180)

                        # --- NEW: Write frame if recording ---
                        # Important: Write the frame *before* rotation or convert the rotated frame back to BGR
                        # Let's write the ORIGINAL BGR frame as captured
                        if self.is_recording and self.video_writer is not None and self.video_writer.isOpened():
                            try:
                                self.video_writer.write(frame_bgr) # Write the original BGR frame
                            except Exception as write_e:
                                print(f"Error writing frame for {self.camera_id}: {write_e}", file=sys.stderr)
                                # Consider stopping recording on write error?
                                # self.stop_recording()
                                # self.error_signal.emit(self.camera_id, f"Video write error: {write_e}")

                        # --- Emit the potentially rotated RGB frame for display ---
                        self.frame_signal.emit(frame_rgb)

                    else:
                        print(f"Warning: Failed to read frame from {self.camera_id}", file=sys.stderr)
                        # Optional: Attempt to reopen camera?
                        time.sleep(0.5) # Wait before retrying

                    # Optional: Add a small sleep to yield CPU if needed, but cap.read() usually blocks
                    # QThread.msleep(1) # e.g., 1ms sleep

            except Exception as e:
                error_msg = f"Camera {self.camera_id} error: {str(e)}"
                print(error_msg, file=sys.stderr)
                import traceback
                print(traceback.format_exc(), file=sys.stderr) # Print full traceback
                self.error_signal.emit(self.camera_id, str(e))
            finally:
                # --- NEW: Ensure recording stops on exit ---
                if self.is_recording:
                    print(f"Thread for {self.camera_id} ending, ensuring recording is stopped.")
                    self.stop_recording()
                # --- End NEW ---

                if self.cap and self.cap.isOpened():
                    self.cap.release()
                    print(f"Released camera: {self.camera_id}")
                self._is_running = False
                self.finished_signal.emit()
                print(f"Camera thread finished: {self.camera_id}")

    def stop(self):
        print(f"Stopping camera thread: {self.camera_id}")
        self._is_running = False # Set flag first
        # stop_recording will be called in the finally block of run
        

    def show(self):
        self._is_running = True
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            self.error_signal.emit(f"Camera {self.source}", f"Failed to open video source {self.source}")
            self.finished_signal.emit()
            return

        # 计算目标帧间隔（秒）
        target_wait_time = 1.0 / 15

        while self._is_running:
            start_time = time.time()  # 记录循环开始时间
            
            ret, frame = self.cap.read()
            if not ret:
                break  # 视频结束
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_signal.emit(frame_rgb)
            
            # 控制帧率：计算剩余等待时间并休眠
            elapsed_time = time.time() - start_time
            remaining_time = max(target_wait_time - elapsed_time, 0)
            time.sleep(remaining_time)
        
        self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        self.finished_signal.emit()

        self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        self.finished_signal.emit()