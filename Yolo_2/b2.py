import cv2
import math
import tkinter as tk
from tkinter import filedialog
import os
from ultralytics import YOLO


model_path = 'D:/Downloads/Cricket/YOLO/runs/detect/train/weights/best.pt'  # Your trained model


root = tk.Tk()
root.withdraw()  
video_path = filedialog.askopenfilename(title="Select a video file", filetypes=[("MP4 Files", "*.mp4"), ("All Files", "*.*")])

if not video_path:
    print("No video file selected.")
    exit()

# Load model
model = YOLO(model_path)

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    print("Error: Unable to detect FPS.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

bat_path = []  
prev_bat_center = None
initial_bat_position = None
bat_angle = 0  

output_dir = 'D:/Downloads/Cricket/Output_Video'  # Customize your directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_video_path = os.path.join(output_dir, 'bat_swing_output.mp4')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break

   
    results = model.predict(source=frame, conf=0.5, save=False, verbose=False)

    if len(results) > 0:
        boxes = results[0].boxes.xywh.cpu().numpy()  # [x_center, y_center, width, height]
        classes = results[0].boxes.cls.cpu().numpy()  # Class IDs

        for box, cls in zip(boxes, classes):
            x_center, y_center, w, h = box
            cls = int(cls)

            # Bat Tracking (Class ID 0)
            if cls == 0:  # Bat class ID
                current_bat_center = (int(x_center), int(y_center))

                
                bat_path.append(current_bat_center)

                # Draw bounding box on the bat
                x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
                x2, y2 = int(x_center + w / 2), int(y_center + h / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

                # Draw the bat's travel path with a purple line
                if len(bat_path) > 1:
                    for i in range(1, len(bat_path)):
                        cv2.line(frame, bat_path[i-1], bat_path[i], (255, 0, 255), 10)  

                
                if prev_bat_center is not None:
                    dx = current_bat_center[0] - prev_bat_center[0]
                    dy = current_bat_center[1] - prev_bat_center[1]
                    angle = math.atan2(dy, dx) * 180 / math.pi  

                    
                    if initial_bat_position is None:
                        initial_bat_position = current_bat_center
                    total_dx = current_bat_center[0] - initial_bat_position[0]
                    total_dy = current_bat_center[1] - initial_bat_position[1]
                    bat_angle = math.atan2(total_dy, total_dx) * 180 / math.pi  

                
                    cv2.putText(frame, f"Swing Angle: {angle:.2f}°", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    
                    cv2.putText(frame, f"Total Travel Angle: {bat_angle:.2f}°", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                prev_bat_center = current_bat_center

    
    cv2.imshow('Bat Swing Detection', frame)

    
    out.write(frame)

   
    if cv2.getWindowProperty('Bat Swing Detection', cv2.WND_PROP_VISIBLE) < 1:
        print("Video window was closed.")
        break


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()


print(f"✅ Bat Swing motion, travel path, and angle detection done! Output saved at: {output_video_path}")
