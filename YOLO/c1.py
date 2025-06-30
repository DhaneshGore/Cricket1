import cv2
import os
from tkinter import Tk, filedialog
from ultralytics import YOLO

# === Configuration ===
model_path = 'D:/Downloads/Cricket/YOLO/runs/detect/train/weights/best.pt'
confidence_threshold = 0.5
ball_class_id = 1
bat_class_id = 0

Tk().withdraw()
video_path = filedialog.askopenfilename(title="Select a Video File",
                                        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
if not video_path:
    print("No video selected. Exiting...")
    exit()

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if width == 0 or height == 0:
    print("Error: Could not read video or video is empty.")
    exit()

# === Prepare Output Path ===
os.makedirs("assets", exist_ok=True)
output_video_path = "assets/whatif_play.mp4"

model = YOLO(model_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

ball_path_real = []
counterfactuals = {
    'fast': [], 'faster': [], 'slow': [], 'slower': []
}
speed_factors = {
    'fast': 1.10, 'faster': 1.20, 'slow': 0.90, 'slower': 0.80
}

frame_idx = 0
paused = False

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=confidence_threshold, save=False, verbose=False)

        if len(results) > 0:
            boxes = results[0].boxes.xywh.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            for box, cls in zip(boxes, classes):
                x_center, y_center, w, h = box
                cls = int(cls)
                if cls == ball_class_id:
                    current_center = (int(x_center), int(y_center))
                    ball_path_real.append(current_center)
                    if len(ball_path_real) >= 2:
                        prev_real = ball_path_real[-2]
                        dx = current_center[0] - prev_real[0]
                        dy = current_center[1] - prev_real[1]
                        for key, factor in speed_factors.items():
                            if len(counterfactuals[key]) == 0:
                                counterfactuals[key].append(prev_real)
                            last = counterfactuals[key][-1]
                            new_pt = (int(last[0] + dx * factor), int(last[1] + dy * factor))
                            counterfactuals[key].append(new_pt)

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

        # Labels
        cv2.putText(frame, "Real Path (Green)", (30, height - 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Fast (Red)", (30, height - 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Faster (Blue)", (30, height - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "Slow (Yellow)", (30, height - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Slower (Pink)", (30, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        out.write(frame)
        cv2.imshow('What-If Counterfactual Ball Simulation', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):
        paused = not paused
    elif key == ord('q'):
        print("Exiting video...")
        break

    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ What-If video saved at: {output_video_path}")
os.system(f'start {output_video_path}')  # Open video in default player (Windows)
