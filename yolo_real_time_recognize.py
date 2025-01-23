import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np

# Constants
YOLO_MODEL_PATH_1 = 'runs/yolo_1/train/weights/best.pt'
YOLO_MODEL_PATH_2 = 'runs/yolo_1/train/weights/best.pt'
YOLO_MODEL_PATH_3 = 'runs/yolo_1/train/weights/best.pt'
YOLO_MODEL_PATH_4 = 'runs/yolo_1/train/weights/best.pt'
YOLO_MODEL_PATH_5 = 'runs/yolo_1/train/weights/best.pt'
YOLO_MODEL_PATH_6 = 'runs/yolo_1/train/weights/best.pt'

CONFIDENCE_THRESHOLD = 0.5
MAX_HANDS = 2 #识别手势的最大数量
HAND_DETECTION_CONFIDENCE = 0.7 #手势识别置信度
CAMERA_INDEX = 0 #相机序号

RECTANGLE_COORDS_1 = [(54, 34), (187, 37), (214, 150), (95, 151)]
RECTANGLE_COORDS_2 = [(130, 202), (274, 204), (284, 298), (158, 295)]
RECTANGLE_COORDS_3 = [(130, 202), (274, 204), (284, 298), (158, 295)]
RECTANGLE_COORDS_4 = [(130, 202), (274, 204), (284, 298), (158, 295)]
RECTANGLE_COORDS_5 = [(130, 202), (274, 204), (284, 298), (158, 295)] #定义零件区域

i=0 # yolo模型序号初始值

# Initialize region
region_1 = False
region_2 = False
region_3 = False
region_4 = False
region_5 = False # 接触器初始值

stage = 'Not start'
assemblyComplete_1 = 0
assemblyComplete_2 = 0
assemblyComplete_3 = 0
assemblyComplete_4 = 0
assemblyComplete_5 = 0 #计数器初始值，只有计数器达到才会触发下一阶段

def initialize_yolo(model_path):
    """Initialize the YOLO model."""
    return YOLO(model_path)

def initialize_mediapipe():
    """Initialize the MediaPipe Hands module."""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False, 
        max_num_hands=MAX_HANDS, 
        min_detection_confidence=HAND_DETECTION_CONFIDENCE
    )
    return hands, mp.solutions.drawing_utils

def is_point_in_rectangle(point, rect_coords):
    """Check if a point is inside the polygon (quadrilateral)."""
    # Using the cv2 function to check if a point is inside the polygon
    return cv2.pointPolygonTest(np.array(rect_coords), point, False) >= 0

def process_frame(frame, yolo_model_group, hands_module, drawing_utils):
    """Process a single video frame with YOLO and MediaPipe."""
    global region_1,stage,assemblyComplete_1,assemblyComplete_2,assemblyComplete_3,i # Use the global region

    # YOLO detection
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
    yolo_results = yolo_model_group[i].predict(source=img_rgb, save=False, conf=0.65, show=False)
    
    # Draw YOLO detection results and display classes in the top-right corner
    yolo_classes = []
    for result in yolo_results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f"{yolo_model_group[i].names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            yolo_classes.append(cls)  # Add class to the list

    # Draw the quadrilateral using cv2.polylines()
    cv2.polylines(frame, [np.array(RECTANGLE_COORDS_1)], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.polylines(frame, [np.array(RECTANGLE_COORDS_2)], isClosed=True, color=(90, 0, 90), thickness=2)
    cv2.polylines(frame, [np.array(RECTANGLE_COORDS_3)], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.polylines(frame, [np.array(RECTANGLE_COORDS_4)], isClosed=True, color=(90, 90, 0), thickness=2)
    cv2.polylines(frame, [np.array(RECTANGLE_COORDS_5)], isClosed=True, color=(0, 0, 255), thickness=2)

    # MediaPipe hand detection
    hand_results = hands_module.process(img_rgb)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Draw hand landmarks
            drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            
            # Check if any hand landmark is inside the polygon
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                if region_1 == False and assemblyComplete_1 < 1:
                    if is_point_in_rectangle((x, y), RECTANGLE_COORDS_1):
                        region_1 = True
                        break  # Increment region_1 only once per hand
                elif region_2 ==False and assemblyComplete_1 >= 1 and assemblyComplete_2 < 1:
                    if is_point_in_rectangle((x, y), RECTANGLE_COORDS_2):
                        region_2 = True
                elif region_3 == False and assemblyComplete_2>=1 and assemblyComplete_3 < 1:
                    if is_point_in_rectangle((x, y), RECTANGLE_COORDS_3):
                        region_3 = True  
                elif region_4 == False and assemblyComplete_3>=1 and assemblyComplete_4 < 1:
                    if is_point_in_rectangle((x, y), RECTANGLE_COORDS_4):
                        region_4 = True 
                elif region_5 == False and assemblyComplete_4>=1 and assemblyComplete_5 < 1:
                    if is_point_in_rectangle((x, y), RECTANGLE_COORDS_5):
                        region_5 = True 
    # Set stage based on region_1 and YOLO class
    if region_1 == True:
        stage = 'Stage 1: 开始记录手势'
        if 1 in yolo_classes and assemblyComplete_1 < 1:
            assemblyComplete_1 += 0.2
        elif 1 in yolo_classes and assemblyComplete_1 >= 1:
            stage = 'Stage 1: 结束记录手势'
            region_1 = False
            i = i + 1
    elif region_2 == True:
        stage = 'Stage 2: 开始记录手势'
        if 1 in yolo_classes and assemblyComplete_2 < 1:
            assemblyComplete_2 += 0.2
        elif 1 in yolo_classes and assemblyComplete_2 >= 1:
            stage = 'Stage 2: 结束记录手势'
            region_2 = False
            i = i + 1
    elif region_3 == True:
        stage = 'Stage 3: 开始记录手势'
        if 1 in yolo_classes and assemblyComplete_3 < 1:
            assemblyComplete_3 += 0.2
        elif 1 in yolo_classes and assemblyComplete_3 >= 1:
            stage = 'Stage 3: 结束记录手势'
            region_3 = False
            i = i + 1
    elif region_4 == True:
        stage = 'Stage 4: 开始记录手势'
        if 1 in yolo_classes and assemblyComplete_4 < 1:
            assemblyComplete_4 += 0.2
        elif 1 in yolo_classes and assemblyComplete_4 >= 1:
            stage = 'Stage 4: 结束记录手势'
            region_4 = False
            i = i + 1
    elif region_5 == True:
        stage = 'Stage 5: 开始记录手势'
        if 1 in yolo_classes and assemblyComplete_5 < 1:
            assemblyComplete_5 += 0.2
        elif 1 in yolo_classes and assemblyComplete_5 >= 1:
            stage = 'Stage 5: 结束记录手势 手势检测结束'
            region_5 = False
         
    # Display the stage in the top-left corner
    cv2.putText(frame, stage, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

def main():
    # Initialize YOLO and MediaPipe
    yolo_model_1 = initialize_yolo(YOLO_MODEL_PATH_1)
    yolo_model_2 = initialize_yolo(YOLO_MODEL_PATH_2)
    yolo_model_3 = initialize_yolo(YOLO_MODEL_PATH_3)
    yolo_model_4 = initialize_yolo(YOLO_MODEL_PATH_4)
    yolo_model_5 = initialize_yolo(YOLO_MODEL_PATH_5)
    yolo_model_6 = initialize_yolo(YOLO_MODEL_PATH_6)
    yolo_model_group = [yolo_model_1, yolo_model_2, yolo_model_3, yolo_model_4, yolo_model_5, yolo_model_6]
    hands_module, drawing_utils = initialize_mediapipe()

    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    # Get video properties for VideoWriter
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use 'XVID' for .avi files
    out = cv2.VideoWriter('output.avi', fourcc, frame_fps, (frame_width, frame_height))

    print("Starting video stream...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Exiting...")
                break

            # Process the frame
            processed_frame = process_frame(frame, yolo_model_group, hands_module, drawing_utils)

            # Write the processed frame to the video file
            out.write(processed_frame)

            # Display the frame
            cv2.imshow("YOLO + MediaPipe Realtime Detection", processed_frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Stream stopped manually.")
    finally:
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Video stream stopped and video saved as 'output.avi'.")

if __name__ == "__main__":
    main()
