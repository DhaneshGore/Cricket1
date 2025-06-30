import cv2
import os
from tkinter import Tk, filedialog
from ultralytics import YOLO

# === Configuration ===
model_path = 'D:/Downloads/Cricket/YOLO/runs/detect/train/weights/best.pt'
confidence_threshold = 0.5
ball_class_id = 1
bat_class_id = 0

# === Select video ===
Tk().withdraw()
video_path = filedialog.askopenfilename(title="🎬 Select a Video File",
                                        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
if not video_path:
    print("❗ No video selected.")
    exit()

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
fps = fps if fps > 0 else 30  # Fallback if FPS not detected

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if width == 0 or height == 0:
    print("❌ Could not read video.")
    exit()

print(f"📂 Video: {video_path}")
print(f"📈 FPS: {fps:.2f}")

# === Output path ===
os.makedirs("assets", exist_ok=True)
output_video_path = "assets/whatif_play.mp4"
print(f"📽️ Output will be saved to: {output_video_path}")

# === Load model ===
model = YOLO(model_path)

# === Video writer ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# === Trajectory Data ===
ball_path_real = []
counterfactuals = {key: [] for key in ['fast', 'faster', 'slow', 'slower']}
speed_factors = {'fast': 1.10, 'faster': 1.20, 'slow': 0.90, 'slower': 0.80}

frame_idx = 0
paused = False

# === Main loop ===
while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=confidence_threshold, save=False, verbose=False)
        if results and results[0].boxes:
            boxes = results[0].boxes.xywh.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            for box, cls in zip(boxes, classes):
                x_center, y_center, w, h = box
                cls = int(cls)

                x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
                x2, y2 = int(x_center + w / 2), int(y_center + h / 2)

                if cls == ball_class_id:
                    current_center = (int(x_center), int(y_center))
                    ball_path_real.append(current_center)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    if len(ball_path_real) >= 2:
                        prev_real = ball_path_real[-2]
                        dx = current_center[0] - prev_real[0]
                        dy = current_center[1] - prev_real[1]

                        for key, factor in speed_factors.items():
                            if len(counterfactuals[key]) == 0:
                                counterfactuals[key].append(prev_real)
                            last_point = counterfactuals[key][-1]
                            new_point = (
                                int(last_point[0] + dx * factor),
                                int(last_point[1] + dy * factor)
                            )
                            counterfactuals[key].append(new_point)

                elif cls == bat_class_id:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # === Draw paths ===
        for i in range(1, len(ball_path_real)):
            cv2.line(frame, ball_path_real[i - 1], ball_path_real[i], (0, 255, 0), 4)

        colors = {
            'fast': (0, 0, 255),
            'faster': (255, 0, 0),
            'slow': (0, 255, 255),
            'slower': (255, 0, 255)
        }

        for key, path in counterfactuals.items():
            for i in range(1, len(path)):
                cv2.line(frame, path[i - 1], path[i], colors[key], 2)

        # === Legend ===
        legends = [
            ("Real Path (Green)", (0, 255, 0)),
            ("Fast (Red)", (0, 0, 255)),
            ("Faster (Blue)", (255, 0, 0)),
            ("Slow (Yellow)", (0, 255, 255)),
            ("Slower (Pink)", (255, 0, 255))
        ]
        y_offset = height - 180
        for text, color in legends:
            cv2.putText(frame, text, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30

        out.write(frame)
        cv2.imshow('🧠 Counterfactual Ball Simulation', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):
        paused = not paused
        print("⏸️ Paused." if paused else "▶️ Resumed.")
    elif key == ord('q'):
        print("🚪 Exiting...")
        break

    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ What-If video saved at: {output_video_path}")
