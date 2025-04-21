# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QComboBox, QDoubleSpinBox,
                             QGridLayout, QFrame, QMessageBox, QFileDialog, QApplication)
from PyQt5.QtCore import Qt, QTimer, QDateTime, pyqtSlot, QDir # Add QDir for directory handling
from PyQt5.QtGui import QPixmap, QImage
import os
import sys
import time
import numpy as np

# --- Imports (keep as they are) ---
try:
    import config
    from camera.opencv_camera import OpenCVCameraThread
    from camera.kinect_camera import KinectCameraThread
    # Check if Kinect is available for conditional logic if needed later
    KINECT_AVAILABLE = KinectCameraThread.PYKINECT2_AVAILABLE # Access the class variable
    from processing.model_handler import ModelHandler
    from processing.scoring import call_backend_scoring
    from utils.image_converter import convert_cv_qt
except ImportError as e:
    print(f"Error importing project modules: {e}", file=sys.stderr)
    print("Please ensure config.py and camera/, processing/, utils/ folders with necessary files exist.", file=sys.stderr)
    sys.exit(1)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV (cv2) not found. Video recording will be disabled.", file=sys.stderr)


class VisualInspectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(config.WINDOW_TITLE)
        self.setGeometry(100, 100, config.INITIAL_WIDTH, config.INITIAL_HEIGHT)

        # --- State variables ---
        self.is_running = False
        self.is_processing = False
        self.current_model = config.DEFAULT_MODEL
        self.current_confidence = config.DEFAULT_CONFIDENCE
        self._is_stopping = False

        # --- Thread instances ---
        self.thread_cam1 = None
        self.thread_cam2 = None
        self.thread_kinect = None

        # --- Latest frames ---
        self.latest_frame_cam1 = None
        self.latest_frame_cam2 = None
        self.latest_frame_kinect = None

        # --- Processing & Scoring ---
        self.model_handler = ModelHandler()
        self.last_score = "N/A"
        self.processing_results = {'cam1': None, 'cam2': None, 'kinect': None}

        # --- UI Elements (keep as they are) ---
        self.start_end_button = None
        self.process_button = None
        self.model_combo = None
        self.confidence_spinbox = None
        self.video_player1 = None
        self.video_player2 = None
        self.video_player_kinect = None
        self.score_display_label = None
        self.status_bar = self.statusBar()

        # --- NEW: Recording state ---
        self.is_recording_active = False # Overall recording state based on Kinect
        self.recordings_folder = "recordings" # Folder name
        self.cam1_video_filename = None
        self.cam2_video_filename = None
        # --- End NEW ---

        # --- Initialization ---
        self.initUI()
        self._ensure_recordings_folder() # Ensure the folder exists on startup
        print("Application Initialized.")
        self.status_bar.showMessage("Ready.")

    # --- NEW: Ensure recordings folder exists ---
    def _ensure_recordings_folder(self):
        """Creates the recordings folder if it doesn't exist."""
        folder_path = self.recordings_folder
        if not QDir().exists(folder_path):
            print(f"Creating recordings folder: {folder_path}")
            if not QDir().mkpath(folder_path):
                 print(f"Error: Could not create recordings folder at {folder_path}", file=sys.stderr)
                 QMessageBox.warning(self, "Folder Creation Error", f"Could not create the folder '{folder_path}'. Recordings might fail.")
                 # Disable recording functionality maybe? Or just warn.
        # else:
            # print(f"Recordings folder found: {folder_path}")


    def initUI(self):
        """Sets up the main UI layout and widgets."""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- 1. Control Area (Layout remains the same) ---
        control_layout = QHBoxLayout()
        # ... (add widgets as before: Model Label, Combo, Confidence Label, Spinbox, Stretch, Start/End, Process) ...
        control_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(config.MODEL_OPTIONS)
        self.model_combo.setCurrentText(config.DEFAULT_MODEL)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        control_layout.addWidget(self.model_combo)

        control_layout.addSpacing(20)
        control_layout.addWidget(QLabel("Confidence:"))
        self.confidence_spinbox = QDoubleSpinBox()
        self.confidence_spinbox.setRange(0.0, 1.0)
        self.confidence_spinbox.setSingleStep(config.CONFIDENCE_STEP)
        self.confidence_spinbox.setValue(config.DEFAULT_CONFIDENCE)
        self.confidence_spinbox.valueChanged.connect(self.on_confidence_changed)
        control_layout.addWidget(self.confidence_spinbox)

        control_layout.addStretch(1)

        self.start_end_button = QPushButton("Start")
        self.start_end_button.setStyleSheet("background-color: lightgreen;")
        self.start_end_button.clicked.connect(self.toggle_stream)
        control_layout.addWidget(self.start_end_button)

        self.process_button = QPushButton("Process")
        self.process_button.setEnabled(False)
        self.process_button.clicked.connect(self.toggle_processing)
        control_layout.addWidget(self.process_button)

        # --- 2. Video Display Area (Layout remains the same) ---
        video_layout = QGridLayout()
        # ... (add video players as before) ...
        video_layout.setRowStretch(0, 2)
        video_layout.setRowStretch(1, 1)
        video_layout.setColumnStretch(0, 1)
        video_layout.setColumnStretch(1, 1)

        self.video_player_kinect = QLabel("Camera 3 (Kinect)")
        self.video_player_kinect.setAlignment(Qt.AlignCenter)
        self.video_player_kinect.setFrameShape(QFrame.Box)
        self.video_player_kinect.setMinimumSize(640, 480)
        self.video_player_kinect.setStyleSheet("background-color: black; color: grey;")
        video_layout.addWidget(self.video_player_kinect, 0, 0, 1, 2)

        self.video_player1 = QLabel("Camera 1 (OpenCV)")
        self.video_player1.setAlignment(Qt.AlignCenter)
        self.video_player1.setFrameShape(QFrame.Box)
        self.video_player1.setMinimumSize(320, 280)
        self.video_player1.setStyleSheet("background-color: black; color: grey;")
        video_layout.addWidget(self.video_player1, 1, 0)

        self.video_player2 = QLabel("Camera 2 (OpenCV)")
        self.video_player2.setAlignment(Qt.AlignCenter)
        self.video_player2.setFrameShape(QFrame.Box)
        self.video_player2.setMinimumSize(320, 280)
        self.video_player2.setStyleSheet("background-color: black; color: grey;")
        video_layout.addWidget(self.video_player2, 1, 1)

        # --- 3. Score Display Area (Layout remains the same) ---
        score_layout = QHBoxLayout()
        # ... (add score label as before) ...
        score_layout.addWidget(QLabel("Processing Score:"))
        self.score_display_label = QLabel("N/A")
        self.score_display_label.setStyleSheet("font-weight: bold; font-size: 16px; color: blue;")
        score_layout.addWidget(self.score_display_label)
        score_layout.addStretch(1)


        # --- Assemble Main Layout ---
        main_layout.addLayout(control_layout)
        main_layout.addLayout(video_layout, stretch=1) # Give video area more stretch
        main_layout.addLayout(score_layout)
        
        # 控制区域添加新按钮
        control_layout.addWidget(QLabel("Model:"))

        self.load_video_button = QPushButton("Load Videos")
        self.load_video_button.clicked.connect(self.load_videos)
        control_layout.addWidget(self.load_video_button)

        print("UI Initialized.")


    # --- Slot functions (on_model_changed, on_confidence_changed remain the same) ---
    def on_model_changed(self, text):
        """Handles model selection changes."""
        self.current_model = text
        print(f"Model selected: {self.current_model}")
        self.status_bar.showMessage(f"Model set to {self.current_model}. Load will occur on processing start.")
        # ... (rest of the logic) ...
        if self.is_processing:
            print("Reloading model for active processing...")
            if not self.model_handler.load_model(self.current_model):
                QMessageBox.warning(self, "Model Load Error", f"Failed to load {self.current_model}")
                self.toggle_processing() # Stop processing if model fails


    def on_confidence_changed(self, value):
        """Handles confidence threshold changes."""
        self.current_confidence = value
        print(f"Confidence threshold set to: {self.current_confidence:.2f}")
        self.status_bar.showMessage(f"Confidence set to {self.current_confidence:.2f}")

    def start_stream_logic(self):
        """Starts the camera threads."""
        if not CV2_AVAILABLE:
             QMessageBox.warning(self, "Dependency Error", "OpenCV (cv2) is not available. Cannot start cameras properly.")
             # return False # Or allow Kinect-only? Depends on desired behavior. For recording, OpenCV is needed.
             # Let's prevent starting if OpenCV isn't there for recording.
             self.status_bar.showMessage("Error: OpenCV required for camera streaming.", 5000)
             return False

        self.status_bar.showMessage("Starting cameras...")
        print("Starting camera threads...")
        self.latest_frame_cam1 = None
        self.latest_frame_cam2 = None
        self.latest_frame_kinect = None
        self.processing_results = {'cam1': None, 'cam2': None, 'kinect': None}
        self._is_stopping = False
        self.is_recording_active = False # Reset recording state
        self.cam1_video_filename = None
        self.cam2_video_filename = None


        cam_started_count = 0
        try:
            # --- Start OpenCV Camera 1 ---
            print(f"Attempting to start OpenCV Camera {config.OPENCV_CAMERA_1_ID}...")
            self.thread_cam1 = OpenCVCameraThread(config.OPENCV_CAMERA_1_ID)
            self.thread_cam1.frame_signal.connect(self.update_frame_cam1)
            self.thread_cam1.error_signal.connect(self.handle_camera_error)
            self.thread_cam1.finished_signal.connect(self.thread_finished)
            # NEW: Connect recording status signal (optional, for UI feedback)
            self.thread_cam1.recording_status_signal.connect(self.handle_recording_status_update)
            self.thread_cam1.start()
            cam_started_count += 1

            # --- Start OpenCV Camera 2 ---
            print(f"Attempting to start OpenCV Camera {config.OPENCV_CAMERA_2_ID}...")
            self.thread_cam2 = OpenCVCameraThread(config.OPENCV_CAMERA_2_ID)
            self.thread_cam2.frame_signal.connect(self.update_frame_cam2)
            self.thread_cam2.error_signal.connect(self.handle_camera_error)
            self.thread_cam2.finished_signal.connect(self.thread_finished)
            # NEW: Connect recording status signal
            self.thread_cam2.recording_status_signal.connect(self.handle_recording_status_update)
            self.thread_cam2.start()
            cam_started_count += 1

            # --- Start Kinect Camera ---
            # Only if available (based on check during import)
            if KINECT_AVAILABLE:
                 print("Attempting to start Kinect Camera...")
                 self.thread_kinect = KinectCameraThread()
                 self.thread_kinect.frame_signal.connect(self.update_frame_kinect)
                 self.thread_kinect.error_signal.connect(self.handle_camera_error)
                 self.thread_kinect.finished_signal.connect(self.thread_finished)
                 if hasattr(self.thread_kinect, 'status_signal'):
                     self.thread_kinect.status_signal.connect(self.update_kinect_status)
                 if hasattr(self.thread_kinect, 'request_app_stop'):
                     self.thread_kinect.request_app_stop.connect(self.handle_kinect_stop_request)

                 # --- *** CONNECT KINECT RECORDING SIGNAL *** ---
                 if hasattr(self.thread_kinect, 'recording_state_signal'):
                      print("Connecting Kinect recording_state_signal...")
                      self.thread_kinect.recording_state_signal.connect(self.handle_recording_state_change)
                 else:
                      print("Warning: Kinect thread does not have 'recording_state_signal'. Recording trigger disabled.")
                 # -----------------------------------------------

                 self.thread_kinect.start()
                 cam_started_count += 1
            elif not KINECT_AVAILABLE:
                 print("Kinect hardware/library not available. Skipping Kinect thread.")
                 self.video_player_kinect.setText("Kinect N/A")
                 # Decide if the app should run without Kinect? Yes, for now.
            else: # Should not happen if PYKINECT2_AVAILABLE was set correctly
                 print("Error determining Kinect availability.")
                 raise RuntimeError("Kinect availability check failed.")


            # Check if at least OpenCV cameras started (needed for core functionality)
            if self.thread_cam1 is None or self.thread_cam2 is None:
                 raise RuntimeError("Failed to start required OpenCV cameras.")

            self.status_bar.showMessage("Cameras starting...")
            return True

        except Exception as e:
            QMessageBox.critical(self, "Camera Start Error", f"Failed to start cameras: {e}\n\nPlease check connections and drivers.")
            print(f"Error during camera startup: {e}", file=sys.stderr)
            import traceback
            print(traceback.format_exc(), file=sys.stderr)
            # Attempt to clean up any threads that might have started
            self.stop_stream_logic() # Call stop logic to clean up partially started threads
            return False


    def stop_stream_logic(self):
        """Stops all camera threads and resets UI."""
        if self._is_stopping:
            print("Already in stopping process, ignoring request.")
            return
        self._is_stopping = True

        self.status_bar.showMessage("Stopping cameras...")
        print("Stopping camera threads...")

        # --- NEW: Ensure recording is stopped before stopping threads ---
        if self.is_recording_active:
             print("Ensuring OpenCV recordings are stopped before thread exit...")
             if self.thread_cam1 and hasattr(self.thread_cam1, 'stop_recording'):
                 self.thread_cam1.stop_recording()
             if self.thread_cam2 and hasattr(self.thread_cam2, 'stop_recording'):
                 self.thread_cam2.stop_recording()
             self.is_recording_active = False # Update state flag
        # --- End NEW ---

        # Request threads to stop
        threads_to_stop = []
        if self.thread_cam1: threads_to_stop.append(self.thread_cam1); print("Requesting stop for Cam1...")
        if self.thread_cam2: threads_to_stop.append(self.thread_cam2); print("Requesting stop for Cam2...")
        if self.thread_kinect: threads_to_stop.append(self.thread_kinect); print("Requesting stop for Kinect...")

        for thread in threads_to_stop:
             if hasattr(thread, 'stop'):
                 thread.stop()

        # Wait for threads to finish
        wait_timeout = 2000 # Increase timeout slightly if needed
        print(f"Waiting up to {wait_timeout}ms for threads to finish...")
        all_stopped = True
        for thread in threads_to_stop:
            if thread.isRunning():
                if not thread.wait(wait_timeout):
                     print(f"Warning: Thread {type(thread).__name__} did not finish within timeout.", file=sys.stderr)
                     all_stopped = False
            # else: # Optional: Check if already finished before wait
                 # print(f"Thread {type(thread).__name__} already finished.")


        if all_stopped:
            print("All threads finished gracefully.")
        else:
             print("Warning: Some threads may not have stopped cleanly.", file=sys.stderr)
             # Consider more forceful termination if required, but avoid if possible


        # Clear thread references
        self.thread_cam1 = None
        self.thread_cam2 = None
        self.thread_kinect = None
        print("Thread references cleared.")

        # Reset UI elements
        for player, text in zip(
            [self.video_player1, self.video_player2, self.video_player_kinect],
            ["Camera 1 (OpenCV)", "Camera 2 (OpenCV)", "Camera 3 (Kinect)" if KINECT_AVAILABLE else "Kinect N/A"]
        ):
            if player:
                player.clear()
                player.setStyleSheet("background-color: black; color: grey;")
                player.setText(text)
        if self.score_display_label: self.score_display_label.setText("N/A")
        self.status_bar.showMessage("Stopped.")

        # Reset recording state variables
        self.is_recording_active = False
        self.cam1_video_filename = None
        self.cam2_video_filename = None

        self._is_stopping = False # Reset stopping flag


    def toggle_stream(self):
        """Toggles the camera stream start/stop state."""
        if self._is_stopping: return # Prevent action if already stopping

        if not self.is_running:
            # --- Start Stream ---
            if self.start_stream_logic():
                self.is_running = True
                self.start_end_button.setText("End")
                self.start_end_button.setStyleSheet("background-color: lightcoral;")
                self.process_button.setEnabled(True) # Enable process button only if stream starts
            else:
                # start_stream_logic already showed error message
                self.is_running = False # Ensure state is false
                self.start_end_button.setText("Start")
                self.start_end_button.setStyleSheet("background-color: lightgreen;")
                self.process_button.setEnabled(False) # Keep process disabled
        else:
            # --- End Stream ---
            # Prevent ending if currently processing? Optional, depends on desired workflow.
            # if self.is_processing:
            #     QMessageBox.warning(self, "Action Required", "Please stop processing before ending the stream.")
            #     return

            self.is_running = False # Set flag first
            if self.is_processing:
                self.toggle_processing() # Stop processing if it's running

            self.stop_stream_logic() # Call the cleanup logic

            # Update button state (should be handled by stop_stream_logic, but double-check)
            self.start_end_button.setText("Start")
            self.start_end_button.setStyleSheet("background-color: lightgreen;")
            self.process_button.setEnabled(False) # Disable process when stream is off


    # toggle_processing remains largely the same
    def toggle_processing(self):
        """Toggles the processing state."""
        if not self.is_processing:
            # --- Start Processing ---
            if not self.is_running:
                QMessageBox.warning(self, "Warning", "Start stream before processing.")
                return

            # Model loading logic...
            self.status_bar.showMessage(f"Loading model {self.current_model}...")
            QApplication.processEvents() # Allow UI to update
            if not self.model_handler.load_model(self.current_model):
                 QMessageBox.critical(self, "Model Error", f"Failed to load model: {self.current_model}")
                 self.status_bar.showMessage(f"Failed to load {self.current_model}.", 5000)
                 return

            print("Starting real-time processing...")
            self.is_processing = True
            self.process_button.setText("Stop Process")
            self.process_button.setStyleSheet("background-color: orange;")
            self.status_bar.showMessage(f"Processing started with {self.current_model}.")
        else:
            # --- Stop Processing ---
            print("Stopping real-time processing...")
            self.is_processing = False
            self.process_button.setText("Process")
            self.process_button.setStyleSheet("") # Reset style
            self.score_display_label.setText("N/A")
            self.status_bar.showMessage("Processing stopped.")
            # Optional: Clear processing results to remove boxes from display immediately
            self.processing_results = {'cam1': None, 'cam2': None, 'kinect': None}
            self.update_display(self.video_player1, self.latest_frame_cam1)
            self.update_display(self.video_player2, self.latest_frame_cam2)
            self.update_display(self.video_player_kinect, self.latest_frame_kinect)


    # update_display remains the same
    def update_display(self, player_label: QLabel, frame: np.ndarray | None, results: list | None = None):
         """Updates a QLabel with a new frame, optionally drawing results."""
         if frame is None:
             # Optionally clear the label or show a placeholder if frame becomes None
             # player_label.clear()
             # player_label.setText("No Signal")
             return

         display_frame = frame.copy() # Work on a copy

         # Draw results if processing and results are available for this frame source
         if self.is_processing and results:
             display_frame = self.model_handler.draw_results(display_frame, results) # Assumes draw_results handles RGB input/output

         # Convert and display
         qimage = convert_cv_qt(display_frame) # Expects RGB
         if qimage:
             pixmap = QPixmap.fromImage(qimage)
             # Scale pixmap smoothly, keeping aspect ratio, fitting the label size
             scaled_pixmap = pixmap.scaled(player_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
             player_label.setPixmap(scaled_pixmap)
         else:
             player_label.setText("Frame Error")
             player_label.setStyleSheet("background-color: red; color: white;")


    # --- Frame Update Slots (remove VideoWriter logic here) ---
    @pyqtSlot(object)
    def update_frame_cam1(self, frame_rgb):
        self.latest_frame_cam1 = frame_rgb
        # Update display with potential processing results
        self.update_display(self.video_player1, frame_rgb, self.processing_results.get('cam1'))
        self.check_and_process_frames() # Check if ready to process

    @pyqtSlot(object)
    def update_frame_cam2(self, frame_rgb):
        self.latest_frame_cam2 = frame_rgb
        self.update_display(self.video_player2, frame_rgb, self.processing_results.get('cam2'))
        self.check_and_process_frames()

    @pyqtSlot(object)
    def update_frame_kinect(self, frame_rgb):
        self.latest_frame_kinect = frame_rgb
        self.update_display(self.video_player_kinect, frame_rgb, self.processing_results.get('kinect'))
        self.check_and_process_frames()

    # --- Processing Logic (check_and_process_frames remains the same) ---
    def check_and_process_frames(self):
        """Checks if all frames are ready and runs processing if enabled."""
        # Check if processing is active AND if all necessary frames have arrived
        # Adapt this check if Kinect is optional
        frames_ready = (self.latest_frame_cam1 is not None and
                        self.latest_frame_cam2 is not None and
                        (self.latest_frame_kinect is not None if KINECT_AVAILABLE else True)) # Only require Kinect frame if available

        if self.is_processing and frames_ready:

            # --- Grab frames for this processing cycle ---
            # Important: Copy the frames to avoid issues if new frames arrive while processing
            frame1_to_process = self.latest_frame_cam1.copy()
            frame2_to_process = self.latest_frame_cam2.copy()
            framek_to_process = self.latest_frame_kinect.copy() if KINECT_AVAILABLE and self.latest_frame_kinect is not None else None

            # --- Clear the latest frame variables immediately ---
            # This signifies that we need new frames for the next cycle
            self.latest_frame_cam1 = None
            self.latest_frame_cam2 = None
            if KINECT_AVAILABLE: self.latest_frame_kinect = None

            # --- Run Inference ---
            start_time = time.perf_counter()
            # Note: model_handler expects RGB frames if it uses libraries like PIL or doesn't convert internally
            results1 = self.model_handler.run_inference(frame1_to_process, self.current_model, self.current_confidence)
            results2 = self.model_handler.run_inference(frame2_to_process, self.current_model, self.current_confidence)
            # Only process Kinect frame if it's available
            resultsk = self.model_handler.run_inference(framek_to_process, self.current_model, self.current_confidence) if framek_to_process is not None else None
            inference_time = time.perf_counter() - start_time

            # Store results for display update
            self.processing_results['cam1'] = results1
            self.processing_results['cam2'] = results2
            if KINECT_AVAILABLE: self.processing_results['kinect'] = resultsk

            # --- Update Display with Processed Frames ---
            # The update_display function will now use these stored results
            self.update_display(self.video_player1, frame1_to_process, results1)
            self.update_display(self.video_player2, frame2_to_process, results2)
            if framek_to_process is not None:
                 self.update_display(self.video_player_kinect, framek_to_process, resultsk)

            # --- Call Scoring ---
            # Adapt scoring call if Kinect results might be None
            score = call_backend_scoring(
                results1, results2, resultsk, # Pass potentially None resultsk
                self.current_model, self.current_confidence
            )
            end_time = time.perf_counter()
            processing_cycle_time = end_time - start_time

            # --- Update Score Display and Status Bar ---
            if score is not None:
                try:
                    self.last_score = f"{float(score):.2f}"
                    self.score_display_label.setText(self.last_score)
                    self.status_bar.showMessage(f"Processed. Score: {self.last_score} (Infer: {inference_time:.3f}s, Total: {processing_cycle_time:.3f}s)")
                except (ValueError, TypeError):
                    self.last_score = "Invalid Score"
                    self.score_display_label.setText(self.last_score)
                    self.status_bar.showMessage(f"Processed with invalid score format '{score}'. (Cycle: {processing_cycle_time:.3f}s)")
            else:
                self.score_display_label.setText("Error")
                self.status_bar.showMessage(f"Scoring error or N/A. (Cycle: {processing_cycle_time:.3f}s)")


    # --- *** NEW: Slot to handle Kinect recording state change *** ---
    @pyqtSlot(bool)
    def handle_recording_state_change(self, should_record):
        """Starts or stops recording on OpenCV cameras based on Kinect signal."""
        if not CV2_AVAILABLE:
             # print("Debug: Received recording signal but OpenCV not available.")
             return # Cannot record without OpenCV

        if should_record and not self.is_recording_active:
             # --- Start Recording ---
             print("Signal received: Start recording on OpenCV cameras.")
             self.is_recording_active = True
             timestamp = QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss")
             self._ensure_recordings_folder() # Make sure folder exists just in case

             # Define filenames
             # Use .avi extension consistent with XVID codec in OpenCV thread
             self.cam1_video_filename = os.path.join(self.recordings_folder, f"cam1_{timestamp}.avi")
             self.cam2_video_filename = os.path.join(self.recordings_folder, f"cam2_{timestamp}.avi")

             # Tell threads to start recording
             if self.thread_cam1 and self.thread_cam1.isRunning():
                  if not self.thread_cam1.start_recording(self.cam1_video_filename):
                       QMessageBox.warning(self, "Recording Error", f"Failed to start recording for {self.thread_cam1.camera_id}")
                       # Should we stop the other one if one fails? Or let it continue? Let's stop all.
                       self.handle_recording_state_change(False) # Request stop immediately
                       return
             else:
                  print(f"Warning: Cannot start recording, {config.OPENCV_CAMERA_1_ID} not running.")
                  # Stop all if one isn't running?
                  self.handle_recording_state_change(False)
                  return


             if self.thread_cam2 and self.thread_cam2.isRunning():
                  if not self.thread_cam2.start_recording(self.cam2_video_filename):
                       QMessageBox.warning(self, "Recording Error", f"Failed to start recording for {self.thread_cam2.camera_id}")
                       self.handle_recording_state_change(False)
                       return
             else:
                  print(f"Warning: Cannot start recording, {config.OPENCV_CAMERA_2_ID} not running.")
                  self.handle_recording_state_change(False)
                  return

             self.status_bar.showMessage("Recording started...", 3000)

        elif not should_record and self.is_recording_active:
             # --- Stop Recording ---
             print("Signal received: Stop recording on OpenCV cameras.")
             self.is_recording_active = False

             # Tell threads to stop recording
             if self.thread_cam1 and self.thread_cam1.isRunning():
                  self.thread_cam1.stop_recording()
             if self.thread_cam2 and self.thread_cam2.isRunning():
                  self.thread_cam2.stop_recording()

             self.status_bar.showMessage("Recording stopped.", 3000)
             # Filenames are reset inside the stop_recording status handler or here if needed
             # print(f"Cam 1 file (intended): {self.cam1_video_filename}")
             # print(f"Cam 2 file (intended): {self.cam2_video_filename}")
             # self.cam1_video_filename = None # Clear reference
             # self.cam2_video_filename = None

    # --- NEW: Slot to update UI based on thread recording status (Optional) ---
    @pyqtSlot(str, bool)
    def handle_recording_status_update(self, camera_id, is_recording):
        """Updates UI elements based on individual camera recording status."""
        print(f"UI Received Recording Status: {camera_id} is recording: {is_recording}")
        # Example: Change border color of the video player
        player = None
        if camera_id == self.thread_cam1.camera_id: # Use optional chaining if thread might be None
             player = self.video_player1
        elif camera_id == self.thread_cam2.camera_id:
             player = self.video_player2

        if player:
             if is_recording:
                 player.setStyleSheet("background-color: black; color: grey; border: 3px solid red;")
             else:
                 player.setStyleSheet("background-color: black; color: grey; border: 1px solid black;") # Reset border


    # --- Kinect Status Update Slot (keep as is) ---
    @pyqtSlot(str)
    def update_kinect_status(self, status_message):
        """Updates the status bar with messages from the Kinect thread."""
        self.status_bar.showMessage(f"Kinect: {status_message}", 5000)

    # --- Kinect Stop Request Slot (keep as is) ---
    @pyqtSlot()
    def handle_kinect_stop_request(self):
        """Handles the stop request signal from the Kinect thread."""
        print("Received stop request from Kinect thread via signal.")
        if self.is_running and not self._is_stopping:
            print("Triggering application stop via toggle_stream()...")
            # Use QTimer.singleShot to avoid issues if the signal comes from a different thread context
            # than the main GUI thread, although PyQt signals are usually thread-safe.
            QTimer.singleShot(0, self.toggle_stream)
        elif self._is_stopping:
            print("Application is already stopping.")
        else:
            print("Application is not running.")

    # --- Error and Cleanup (handle_camera_error, thread_finished, closeEvent) ---
    # Modify handle_camera_error to potentially stop recording if an error occurs
    @pyqtSlot(str, str)
    def handle_camera_error(self, camera_id, error_message):
        print(f"Error signal received from {camera_id}: {error_message}", file=sys.stderr)
        # Prevent multiple popups during shutdown
        if not self._is_stopping:
             QMessageBox.warning(self, "Camera Error", f"Camera {camera_id} reported an error:\n{error_message}")
             self.status_bar.showMessage(f"ERROR in {camera_id}! Attempting to stop.", 10000)
             # If a camera fails, stop the whole stream and potentially recording
             if self.is_running:
                 # Use QTimer to ensure it runs in the main thread context after signal handling
                 QTimer.singleShot(0, self.toggle_stream) # Graceful stop using the toggle

    @pyqtSlot()
    def thread_finished(self):
        sender_thread = self.sender()
        thread_name = "Unknown Thread"
        if sender_thread:
            thread_name = type(sender_thread).__name__
            print(f"Thread finished signal received from: {thread_name} (ID: {sender_thread})")
            # References are cleared in stop_stream_logic, no need to clear here usually
            # Optional: Check if the finished thread is one we still hold a reference to
            # if sender_thread == self.thread_cam1: self.thread_cam1 = None
            # elif sender_thread == self.thread_cam2: self.thread_cam2 = None
            # elif sender_thread == self.thread_kinect: self.thread_kinect = None
        else:
            print("Thread finished signal received, but sender is None.")


    def closeEvent(self, event):
        print("Close event triggered. Cleaning up...")
        if self._is_stopping:
             print("Close event ignored, already stopping.")
             event.ignore() # Should ideally accept if stopping initiated by close
             return

        reply = QMessageBox.question(self, 'Confirm Exit',
                                     "Are you sure you want to exit?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self._is_stopping = True # Set flag immediately
            self.status_bar.showMessage("Closing application...")
            self.is_running = False # Ensure state flags reflect intent
            self.is_processing = False

            self.stop_stream_logic() # Call the main stopping logic

            print("Exiting application.")
            event.accept() # Accept the close event
        else:
            print("Close cancelled.")
            self.status_bar.showMessage("Ready.")
            event.ignore() # Ignore the close event
            
    def load_videos(self):
        if self.is_running:
            self.toggle_stream()  # 停止当前摄像头流

        # 选择视频文件
        cam1_video = "video\cam1.avi"
        cam2_video = "video\cam2.avi"
        kinect_video = "video\kinect.avi"

        if not cam1_video or not cam2_video or not kinect_video:
            QMessageBox.warning(self, "Error", "All three videos must be selected.")
            return

        # 创建视频播放线程
        self.thread_cam1 = OpenCVCameraThread(cam1_video)
        self.thread_cam1.frame_signal.connect(self.update_frame_cam1)
        self.thread_cam1.error_signal.connect(self.handle_camera_error)
        self.thread_cam1.finished_signal.connect(self.thread_finished)
        self.thread_cam1.start()

        self.thread_cam2 = OpenCVCameraThread(cam2_video)
        self.thread_cam2.frame_signal.connect(self.update_frame_cam2)
        self.thread_cam2.error_signal.connect(self.handle_camera_error)
        self.thread_cam2.finished_signal.connect(self.thread_finished)
        self.thread_cam2.start()

        if KINECT_AVAILABLE:
            self.thread_kinect = OpenCVCameraThread(kinect_video)
            self.thread_kinect.frame_signal.connect(self.update_frame_kinect)
            self.thread_kinect.error_signal.connect(self.handle_camera_error)
            self.thread_kinect.finished_signal.connect(self.thread_finished)
            self.thread_kinect.start()
        else:
            self.video_player_kinect.setText("Kinect N/A")

        # 更新状态
        self.is_running = True
        self.start_end_button.setText("End")
        self.start_end_button.setStyleSheet("background-color: lightcoral;")
        self.process_button.setEnabled(True)


# --- Main Execution Block (if this is the main script) ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Optional: Set application style, icon, etc.
    # app.setStyle('Fusion')
    main_window = VisualInspectionApp()
    main_window.show()
    sys.exit(app.exec_())