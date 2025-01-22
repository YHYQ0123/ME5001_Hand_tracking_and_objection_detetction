import cv2
from ultralytics import YOLO
import mediapipe as mp

# 加载YOLOv8模型
model_1 = YOLO('../models/model_1_2/best.pt')
model_2 = YOLO('../models/model_2_3/best.pt')
modellist = [model_1, model_2]  # 确保所有模型都在列表中
model_number = 0  # 当前使用的模型编号
model = modellist[model_number]  # 初始化模型

# 初始化 MediaPipe 手部检测模块
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 打开摄像头
cap = cv2.VideoCapture(1)  # 0表示使用默认摄像头

# 获取摄像头的宽度和高度
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# 初始化 VideoWriter 用于录制视频
out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 15, (frame_width, frame_height))

# 初始化计数器
confidence_threshold = 0.88  # 设置置信度阈值
consecutive_frames = 0  # 用于记录连续出现的帧数
required_frames = 8  # 连续8帧满足条件

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 使用YOLO模型进行推理
    results = model(frame)
    
    # 在帧上绘制YOLO检测结果
    annotated_frame = results[0].plot()
    
    found_target = False  # 用于标记是否找到目标对象
    
    # 遍历YOLO检测结果的边界框
    for box in results[0].boxes:
        class_id = int(box.cls[0])  # 获取类别索引
        confidence = box.conf[0]  # 获取置信度
        
        # 如果检测到类别为1且置信度超过阈值
        if class_id == 1 and confidence > confidence_threshold:
            found_target = True  # 标记为找到目标
        
        # 取出边界框的坐标
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # 绘制中心点和中心坐标
        cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)
        cv2.putText(annotated_frame, f"Center: ({center_x}, {center_y})", 
                    (center_x - 50, center_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 判断是否连续检测到目标
    if found_target:
        consecutive_frames += 1
    else:
        consecutive_frames = 0  # 重置计数器
    
    # 如果连续8帧都检测到目标类
    if consecutive_frames >= required_frames and model_number < len(modellist) - 1:
        model_number = (model_number + 1)
        model = modellist[model_number]  # 选择新的模型
        consecutive_frames = 0  # 重置计数器
    
    # BGR转换为RGB（因为MediaPipe使用RGB图像）
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 使用MediaPipe检测手部关键点
    hand_results = hands.process(rgb_frame)
    
    # 如果检测到手部，绘制手部关键点
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )
    
    # 显示带注释的帧
    cv2.imshow('YOLOv8 + MediaPipe', annotated_frame)

    # 将处理后的帧写入视频文件
    out.write(annotated_frame)
    
    # 按下'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和 VideoWriter 资源并关闭窗口
cap.release()
out.release()  # 释放视频写入对象
cv2.destroyAllWindows()