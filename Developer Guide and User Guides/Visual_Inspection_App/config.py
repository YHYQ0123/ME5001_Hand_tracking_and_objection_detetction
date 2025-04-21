# --- General ---
APP_NAME = "Assembly Verification System Host Computer Interface"
CURRENT_DATE = "2025-04-06" # Example, you might want to get this dynamically

# --- Frame Dimensions ---
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# --- Camera Configuration ---
OPENCV_CAMERA_1_ID = 1  # First OpenCV camera ID
OPENCV_CAMERA_2_ID = 2  # Second OpenCV camera ID
# Kinect configuration might be more complex (e.g., device index) - handled in kinect_camera.py

# --- Model Configuration ---
DEFAULT_MODEL = "YOLO_group" # Default model
MODEL_OPTIONS = ["YOLO_single", "YOLO_group"] # Available models
DEFAULT_CONFIDENCE = 0.85 # Default confidence threshold
CONFIDENCE_STEP = 0.05

# --- Recording Configuration ---
RECORDINGS_FOLDER = "recordings"
VIDEO_FORMAT = ".mp4"
RECORDING_FPS = 20.0 # Target FPS for recording (adjust as needed)

# --- UI Configuration ---
WINDOW_TITLE = f"{APP_NAME} (v1.0 - {CURRENT_DATE})"
INITIAL_WIDTH = 1600 # Increased width for potentially wider displays
INITIAL_HEIGHT = 900 # Increased height
MIN_VIDEO_LABEL_WIDTH = 270 # Minimum width for video labels
MIN_VIDEO_LABEL_HEIGHT = 480 # Minimum height (maintaining 16:9 ratio)