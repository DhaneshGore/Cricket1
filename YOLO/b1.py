import torch
import cv2
import numpy as np
import mediapipe as mp
import math
from ultralytics import YOLO

# Initialize MediaPipe pose detector
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load trained YOLO model
model_path = 'D:/Downloads/Cricket/YOLO/runs/detect/train/weights/best.pt'
model = YOLO(model_path)

def calculate_angle(a, b, c):
    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    return abs(angle)

# Load cricket video
video_path = 'D:/Downloads/Cricket/4.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results[0].boxes
    bat_detected = False

    for detection in detections:
        box = detection.xyxy[0].cpu().numpy()
        class_id = int(detection.cls[0].cpu().numpy())

        if class_id == 0:  # Bat class
            bat_detected = True
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if bat_detected:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(image_rgb)

        if results_pose.pose_landmarks:
            lm = results_pose.pose_landmarks.landmark
            wrist = (lm[mp_pose.PoseLandmark.LEFT_WRIST].x * frame.shape[1],
                     lm[mp_pose.PoseLandmark.LEFT_WRIST].y * frame.shape[0])
            elbow = (lm[mp_pose.PoseLandmark.LEFT_ELBOW].x * frame.shape[1],
                     lm[mp_pose.PoseLandmark.LEFT_ELBOW].y * frame.shape[0])
            shoulder = (lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1],
                        lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0])

            actual_angle = calculate_angle(shoulder, elbow, wrist)

            # Draw actual swing angle
            cv2.putText(frame, f'Actual Swing: {int(actual_angle)}°', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Simulate what-if swings (±10, ±20 degrees)
            offsets = [-20, -10, 10, 20]
            y_offset = 60
            for offset in offsets:
                sim_angle = actual_angle + offset
                color = (0, 0, 255) if offset > 0 else (255, 0, 255)
                label = f'What-If: {int(sim_angle)}° ({"+" if offset > 0 else ""}{offset})'
                cv2.putText(frame, label, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                y_offset += 30

    cv2.imshow('Bat Swing + What-If Simulation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
