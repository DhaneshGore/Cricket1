import cv2
import math
import tkinter as tk
from tkinter import filedialog
import os
import sys
import numpy as np
from ultralytics import YOLO

# === Constants ===
BALL_CLASS_ID = 1
BOWLER_CLASS_ID = 3
CONFIDENCE_THRESHOLD = 0.5

# === Load YOLO model ===
model_path = 'Cricket1/YOLO/runs/detect/train/weights/best.pt'
model = YOLO(model_path)

# === Select video file ===
root = tk.Tk()
root.withdraw()
video_path = filedialog.askopenfilename(title="🎬 Select a video file", filetypes=[("MP4 Files", "*.mp4"), ("All Files", "*.*")])

if not video_path:
    print("❗ No video file selected.")
    exit()

# === Setup output path ===
os.makedirs("assets", exist_ok=True)
output_video_path = "assets/actual_play.mp4"

# === Open video ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Error opening video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
fps = fps if fps > 0 else 30  # fallback
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
time_per_frame = 1 / fps

# === Output video writer ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# === Tracking variables ===
ball_path = []
bowler_path = []
prev_ball_center = None
max_ball_speed = 0

print(f"🔍 Processing: {os.path.basename(video_path)}")
print(f"🎯 FPS: {fps:.2f}")

# === Process frames ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("✅ Video processing completed.")
        break

    results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, save=False, verbose=False)

    if results and results[0].boxes:
        boxes = results[0].boxes.xywh.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):
            x_center, y_center, w, h = box
            cls = int(cls)

            x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
            x2, y2 = int(x_center + w / 2), int(y_center + h / 2)

            if cls == BALL_CLASS_ID:
                current_center = (int(x_center), int(y_center))
                ball_path.append(current_center)

                if prev_ball_center is not None:
                    distance = math.hypot(current_center[0] - prev_ball_center[0],
                                          current_center[1] - prev_ball_center[1])
                    speed = distance / time_per_frame
                    max_ball_speed = max(max_ball_speed, speed)

                    cv2.putText(frame, f"Speed: {speed:.2f} px/sec", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                prev_ball_center = current_center
                cv2.rectangle(frame, (x1, y1), (x2, y2), (35, 35, 164), 4)

            elif cls == BOWLER_CLASS_ID:
                bowler_center = (int(x_center), int(y_center))
                bowler_path.append(bowler_center)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)

    # === Draw paths ===
    if len(ball_path) > 2:
        pts = np.array(ball_path, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 0), thickness=4, lineType=cv2.LINE_AA)

    if len(bowler_path) > 2:
        pts = np.array(bowler_path, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=False, color=(255, 0, 0), thickness=4, lineType=cv2.LINE_AA)

    # === Max speed display ===
    cv2.putText(frame, f"Max Ball Speed: {max_ball_speed:.2f} px/sec", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

    # === Show and write frame ===
    cv2.imshow('🎥 Ball & Bowler Tracking', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Stopped by user.")
        break

# === Cleanup ===
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Tracking complete. Output saved to: {output_video_path}")
