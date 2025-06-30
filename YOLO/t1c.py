import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
import sys
import cv2
import numpy as np
import pyautogui
from ultralytics import YOLO

# --- Initialize Tkinter ---
root = tk.Tk()
root.withdraw()

# --- Ask user for detection mode ---
answer = messagebox.askyesno(
    "Select Mode",
    "Yes: Live Screen Detection\nNo: Select Video File\n\nDo you want to run live screen detection?"
)

# --- Prepare output folder ---
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
output_folder = os.path.join(desktop_path, "Vraj_Assignment")
os.makedirs(output_folder, exist_ok=True)

# --- Live Detection using screenshots ---
def detect_on_live_screen():
    print("🟢 Starting live screen detection...")
    model = YOLO("runs/detect/train/weights/best.pt")

    try:
        while True:
            screenshot = pyautogui.screenshot()
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

            # Resize for performance (optional)
            # frame = cv2.resize(frame, (1280, 720))

            results = model.predict(source=frame, conf=0.3, verbose=False)

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        label = model.names[cls]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("🖥️ Live Screen Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 Detection stopped by user.")
                break

    except KeyboardInterrupt:
        print("⛔ Interrupted by user.")
    finally:
        cv2.destroyAllWindows()

# --- Video Detection using YOLO CLI ---
def detect_on_video(file_path):
    code_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    input_filename = os.path.splitext(os.path.basename(file_path))[0]
    save_dir = os.path.join(output_folder, f"{input_filename}_{code_name}")

    command = (
        f"yolo predict model=\"runs/detect/train/weights/best.pt\" "
        f"source=\"{file_path}\" "
        f"save=True save_txt=True "
        f"project=\"{output_folder}\" "
        f"name=\"{input_filename}_{code_name}\" "
        f"exist_ok=True"
    )

    print("🛠️ Running command:\n", command)
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"✅ Predictions saved at: {save_dir}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running YOLO detection:\n{e}")

# --- Run the selected mode ---
if answer:
    detect_on_live_screen()
else:
    file_path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
    )
    if file_path:
        detect_on_video(file_path)
    else:
        print("⚠️ No video file selected.")
