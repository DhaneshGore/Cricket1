import cv2
import os
from tkinter import Tk, filedialog
from ultralytics import YOLO
from datetime import datetime

# Class_id to label for shots
shot_class_names = {
    0: "Drive",
    1: "Defensive",
    2: "Aggressive",
    3: "Leave"
}

# Class_id to label for batsman detection 
batsman_class_names = {
    4: "Batsman"  # Class 4 for batsman detection from model 1
}

# Colors
actual_color = (0, 255, 0)
hypo_color = (255, 0, 0)

# Load YOLO models
shot_model = YOLO("D:/Downloads/Cricket/Yolo_2/runs/detect/train/weights/best.pt")
batsman_model = YOLO("D:/Downloads/Cricket/YOLO/runs/detect/train/weights/best.pt")

# File selection
root = Tk()
root.withdraw()
video_path = filedialog.askopenfilename(title="Select a video file")

# Output setup
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
output_folder = os.path.join(desktop_path, "Vraj_assignment")
os.makedirs(output_folder, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_video_path = os.path.join(output_folder, f"{os.path.basename(video_path).split('.')[0]}_output_{timestamp}.mp4")

# Open video
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

def draw_two_bar_chart(frame, actual_cls, hypo_cls, actual_conf, hypo_conf):
    chart_top = height - 180
    chart_bottom = height - 30
    max_bar_height = chart_bottom - chart_top

    bar_width = 60
    spacing = 200
    start_x = 200

    bars = []

    if actual_cls is not None:
        bars.append({
            "label": f"Actual: {shot_class_names[actual_cls]}",
            "conf": actual_conf,
            "color": actual_color,
            "x": start_x
        })

    if hypo_cls is not None:
        bars.append({
            "label": f"Hypo: {shot_class_names[hypo_cls]}",
            "conf": hypo_conf,
            "color": hypo_color,
            "x": start_x + spacing
        })

    for bar in bars:
        height_val = int(bar["conf"] * max_bar_height)
        x = bar["x"]
        y_top = chart_bottom - height_val
        cv2.rectangle(frame, (x, chart_bottom), (x + bar_width, y_top), bar["color"], -1)
        cv2.putText(frame, bar["label"], (x - 10, chart_bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar["color"], 2)
        cv2.putText(frame, f"{int(bar['conf'] * 100)}%", (x + 5, y_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# Process video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    actual_cls = None
    actual_conf = 0
    hypo_cls = None
    hypo_conf = 0

    # Detect batsman from model 1
    batsman_results = batsman_model(frame)[0]
    batsman_detected = any(int(box.cls[0]) == 4 for box in batsman_results.boxes)

    if batsman_detected:
        # Detect shot
        shot_results = shot_model(frame)[0]
        for box in shot_results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cropped = frame[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            actual_cls = cls_id
            actual_conf = conf

            # Flip for hypothetical
            flipped = cv2.flip(cropped, 1)
            hypo_results = shot_model(flipped)[0]

            if hypo_results.boxes:
                hypo_box = hypo_results.boxes[0]
                hypo_cls = int(hypo_box.cls[0])
                hypo_conf = float(hypo_box.conf[0])

            # Draw shot box
            label_actual = f"Actual: {shot_class_names.get(actual_cls, 'Unknown')} ({actual_conf:.2f})"
            label_hypo = f"Hypo: {shot_class_names.get(hypo_cls, 'Unknown')} ({hypo_conf:.2f})" if hypo_cls is not None else "Hypo: No Prediction"
            cv2.rectangle(frame, (x1, y1), (x2, y2), actual_color, 2)
            cv2.putText(frame, label_actual, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, actual_color, 2)
            cv2.putText(frame, label_hypo, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hypo_color, 2)

            break  # Only first shot considered

    # Draw chart
    draw_two_bar_chart(frame, actual_cls, hypo_cls, actual_conf, hypo_conf)

    out.write(frame)
    cv2.imshow("Batting Style Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video saved at: {output_video_path}")
