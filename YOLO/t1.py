import tkinter as tk
from tkinter import filedialog
import cv2
from ultralytics import YOLO

# === File Picker Dialog ===
root = tk.Tk()
root.withdraw()
video_path = filedialog.askopenfilename(
    title="🎬 Select a video file",
    filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
)

if not video_path:
    print("❗ No video selected.")
    exit()

# === Load YOLO model ===
model = YOLO("D:/Downloads/Cricket/YOLO/runs/detect/train/weights/best.pt")  # Or yolov8n.pt

# === Open selected video ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Failed to open video.")
    exit()

print("✅ Running YOLO on uploaded video. Press 'q' to stop.")

# === Process video frame-by-frame ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict with YOLO
    results = model.predict(source=frame, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()

    # Show result
    cv2.imshow("📹 YOLO Detection on Video", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Detection completed.")
