import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from ultralytics import YOLO
import os

# === Load Models ===
model_ball = YOLO("Cricket1/YOLO/runs/detect/train/weights/best.pt")   # ball model
model_batsman = YOLO("Cricket1/Yolo_3/runs/detect/train/weights/best.pt")  # hand/batsman model

def get_positions(model, frame):
    results = model(frame)
    if hasattr(results[0], 'boxes'):
        boxes = results[0].boxes.xywh.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        return list(zip(boxes, classes, confidences))
    return []

def get_angle(cx, cy, px, py):
    dx, dy = px - cx, py - cy
    return np.arctan2(dy, dx) * 180 / np.pi

def determine_fielder_position(angle):
    if 45 <= angle < 135:
        return "Cover"
    elif -45 <= angle < 45:
        return "Mid-Off"
    elif -135 <= angle < -45:
        return "Slip"
    else:
        return "Leg Slip"

def draw_annotations(frame, positions, center_x, center_y, view='front'):
    for (xywh, class_id, conf) in positions:
        x_center, y_center, w, h = xywh
        x_center, y_center, w, h = int(x_center), int(y_center), int(w), int(h)
        x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
        x2, y2 = int(x_center + w / 2), int(y_center + h / 2)

        angle = get_angle(center_x, center_y, x_center, y_center)
        if view == 'rear':
            angle = -angle

        position_label = determine_fielder_position(angle)
        color = (0, 255, 255) if class_id == 1 else (255, 0, 0)  # color code (e.g., yellow for hand, red for batsman)

        label_text = f"{model_ball.names[int(class_id)]} | {position_label} ({conf:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def process_video(video_path, view='front'):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center_x, center_y = width // 2, height // 2

    output_path = os.path.join(os.path.dirname(video_path), "player_positioning_output.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        positions_ball = get_positions(model_ball, frame)
        positions_hand = get_positions(model_batsman, frame)

        draw_annotations(frame, positions_ball + positions_hand, center_x, center_y, view=view)

        out.write(frame)
        cv2.imshow("Player Positioning", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Output saved at: {output_path}")

def main():
    view = input("View from front or rear? (f/r): ").strip().lower()
    view_type = 'rear' if view == 'r' else 'front'

    Tk().withdraw()
    video_path = askopenfilename(title="Select cricket video", filetypes=[("Video files", "*.mp4 *.mov *.avi")])
    if video_path:
        process_video(video_path, view=view_type)
    else:
        print("❌ No video selected.")

if __name__ == "__main__":
    main()
