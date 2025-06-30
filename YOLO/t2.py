import cv2
import math
import os
from ultralytics import YOLO

# ==== SETTINGS ====
model_path = "D:/Downloads/Cricket/YOLO/runs/detect/train/weights/best.pt"  # Your custom trained model
video_path = "D:/Downloads/Cricket/vk.mp4"  # Input video
output_dir = "assets"
output_video_path = os.path.join(output_dir, "actual_play.mp4")
ball_class_id = 1  # Update if needed
confidence_threshold = 0.5
# ===================

# Create output folder if missing
os.makedirs(output_dir, exist_ok=True)

# Load YOLO model
model = YOLO(model_path)

# Open video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if width == 0 or height == 0:
    print("❌ Error: Could not read video or video is empty.")
    exit()

# Prepare output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Tracking variables
prev_center = None
frame_idx = 0
time_per_frame = 1 / fps

# === Start Frame Processing ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model.predict(source=frame, conf=confidence_threshold, save=False, verbose=False)

    if results and results[0].boxes:
        boxes = results[0].boxes.xywh.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):
            if int(cls) == ball_class_id:
                x_center, y_center, w, h = box
                center = (int(x_center), int(y_center))

                # Draw ball
                cv2.circle(frame, center, 10, (0, 255, 0), -1)

                # Draw trajectory
                if prev_center is not None:
                    cv2.line(frame, prev_center, center, (255, 0, 0), 2)
                    distance = math.hypot(center[0] - prev_center[0], center[1] - prev_center[1])
                    speed = distance / time_per_frame  # px/sec
                    cv2.putText(frame, f"Speed: {speed:.2f} px/s", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                prev_center = center
                break  # track only one ball

    else:
        prev_center = None  # reset if no ball found

    out.write(frame)
    cv2.imshow('🏏 Ball Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Video stopped by user.")
        break

    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Ball tracking video saved to: {output_video_path}")
