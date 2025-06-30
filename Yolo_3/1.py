import cv2
from ultralytics import YOLO
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os

# === Load Two YOLOv8 Models ===
model_path_1 = "D:/Downloads/Cricket/Yolo/runs/detect/train/weights/best.pt"  # 🔁 Replace with your first model path (detecting class 1)
model_path_2 = "D:/Downloads/Cricket/Yolo_3/runs/detect/train/weights/best.pt"  # 🔁 Replace with your second model path (detecting class 1 and 2)


# Load the models
model_1 = YOLO(model_path_1)
model_2 = YOLO(model_path_2)

# === Open File Dialog to Choose Video ===
Tk().withdraw()  # Hide the main tkinter window
video_path = askopenfilename(title="Select a video", filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])

if not video_path:
    print("❌ No video selected.")
    exit()

# === Load Video ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Error opening video.")
    exit()

# === Video Writer Setup ===
output_path = os.path.splitext(video_path)[0] + "_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# === Inference Loop ===
print("🚀 Running inference...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run first model (detect class ID 1 - ball)
    results_1 = model_1(frame)[0]

    # Run second model (detect class ID 1 - right hand, class ID 2 - left hand)
    results_2 = model_2(frame)[0]

    # Detect the ball (class ID 1)
    for box in results_1.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls_id == 1 and conf > 0.5:  # Ball detection (class ID 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)  # Green for ball
            cv2.putText(frame, f"Ball {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 4)

    # Detect right hand (class ID 1) and left hand (class ID 2)
    for box in results_2.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model_2.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls_id == 1 and conf > 0.5:  # Right hand (class ID 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)  # Red for right hand
            cv2.putText(frame, f"Right Hand {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 4)

        elif cls_id == 2 and conf > 0.5:  # Left hand (class ID 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)  # Blue for left hand
            cv2.putText(frame, f"Left Hand {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 4)

    # Write the frame to the output video
    out.write(frame)
    cv2.imshow("YOLOv8 Video Inference", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Inference complete. Output saved to:\n{output_path}")
