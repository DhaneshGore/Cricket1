import cv2
import math
import tkinter as tk
from tkinter import filedialog
import os
import sys
from ultralytics import YOLO

# Load YOLO model
model_path = "D:/Downloads/Cricket/YOLO/yolov8n.pt"
model = YOLO(model_path)

# Select video
root = tk.Tk()
root.withdraw()
video_path = filedialog.askopenfilename(title="Select a video", filetypes=[("MP4 files", "*.mp4")])
if not video_path:
    print("No video selected.")
    exit()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output path setup
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
output_folder = os.path.join(desktop_path, "Vraj_Assignment")
os.makedirs(output_folder, exist_ok=True)

video_name = os.path.splitext(os.path.basename(video_path))[0]
script_name = os.path.splitext(os.path.basename(__file__))[0]
output_filename = f"{video_name}_{script_name}.mp4"
output_video_path = os.path.join(output_folder, output_filename)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Tracking variables
bat_path = []
prev_center = None
initial_pos = None

# Frame processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.5, save=False, verbose=False)

    if len(results) > 0:
        boxes = results[0].boxes.xywh.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):
            x, y, w, h = box
            cls = int(cls)

            if cls == 0:  # Bat
                center = (int(x), int(y))
                bat_path.append(center)

                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

                # Draw actual path
                for i in range(1, len(bat_path)):
                    cv2.line(frame, bat_path[i-1], bat_path[i], (255, 0, 255), 3)  # Purple actual path

                if prev_center is not None:
                    dx = center[0] - prev_center[0]
                    dy = center[1] - prev_center[1]
                    angle = math.atan2(dy, dx) * 180 / math.pi
                    cv2.putText(frame, f"Actual: {angle:.2f}°", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # Simulated what-if angles
                    offsets = [-20, -10, 10, 20]
                    colors = [(255, 0, 0), (255, 100, 100), (0, 255, 255), (0, 100, 255)]
                    y_base = 80
                    for i, offset in enumerate(offsets):
                        sim_angle = angle + offset
                        label = f"What-If {offset:+}: {sim_angle:.2f}°"
                        cv2.putText(frame, label, (50, y_base + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)

                    # Optionally draw ghost lines for each simulated angle
                    r = 60  # distance to draw
                    for i, offset in enumerate(offsets):
                        sim_rad = math.radians(angle + offset)
                        ghost_end = (int(center[0] + r * math.cos(sim_rad)), int(center[1] + r * math.sin(sim_rad)))
                        cv2.line(frame, center, ghost_end, colors[i], 2)

                prev_center = center

    out.write(frame)
    cv2.imshow("Bat Swing Detection + What-If", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Output saved at: {output_video_path}")
