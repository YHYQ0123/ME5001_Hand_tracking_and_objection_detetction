# Hand Tracking and Object Detection for Assembly Process Verification  
[![GitHub](https://img.shields.io/github/license/0123YHYQ1129/ME5001_Hand_tracking_and_objection_detetction)](https://github.com/0123YHYQ1129/ME5001_Hand_tracking_and_objection_detetction)  

## Overview  
This project proposes an **integrated vision-based system** for automated verification of manual assembly processes, combining multi-sensor data fusion, state-aware object detection, and motion similarity assessment. The system ensures procedural correctness in human-centric manufacturing by monitoring operator hand movements and assembly state progression.  

---

## Key Features  
1. **Multi-Sensor Data Acquisition**  
   - **Hybrid Sensor Setup**: Azure Kinect (RGB-D) + dual Logitech RGB cameras for 360° coverage.  
   - **3D Hand Reconstruction**: Fusion of MediaPipe 2D keypoints and depth data via SVD-optimized triangulation.  

2. **State-Aware YOLO Group Model**  
   - **Sliding YOLO Architecture**: Specialized YOLOv8s models for sequential stage triplets, improving classification accuracy for adjacent assembly states.  
   - **Multi-View Consensus**: Synchronized inference across 3 cameras with temporal stability filtering.  

3. **Motion Similarity Evaluation**  
   - **Transformer-Based Siamese Network**: Quantifies similarity between 3D hand kinematic sequences and standard templates (84% accuracy on filtered data).  
   - **Temporal Pooling**: Handles variable-length sequences with dynamic window smoothing.  

4. **PyQt-Based GUI**  
   - Real-time operator guidance, synchronized video display, and validation feedback.  

---

## Technical Highlights  
- **YOLOv8s**: Optimized for real-time assembly state recognition.  
- **Transformer Encoder**: Captures long-range dependencies in motion sequences.  
- **SVD Optimization**: Enhances 3D reconstruction robustness.  

---

## Performance  
| Component                | Accuracy/Improvement                     |  
|--------------------------|------------------------------------------|  
| YOLO Group (Adjacent Stages) | **+12% Precision** vs. Single YOLO       |  
| Motion Similarity Model  | **84% Accuracy** (frame-normalized data) |  

---

## Installation  
```bash  
git clone https://github.com/0123YHYQ1129/ME5001_Hand_tracking_and_objection_detetction.git  
pip install -r requirements.txt  
