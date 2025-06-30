import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from ultralytics import YOLO  # Correct YOLOv8 model loading

# Define paths for the models
model_path_1 = "D:/Downloads/Cricket/YOLO/runs/detect/train/weights/best.pt"   # ball
model_path_2 = "D:/Downloads/Cricket/Yolo_3/runs/detect/train/weights/best.pt" # batsman hand

# Load the models using YOLOv8
model_ball = YOLO(model_path_1)
model_batsman = YOLO(model_path_2)

def get_positions(model, image):
    """
    Runs the model on the given image to detect objects and return positions (bounding boxes).
    
    Args:
    - model: The YOLOv8 model to run inference with.
    - image: Input image (numpy array or path to the image).
    
    Returns:
    - positions: List of detected objects with their bounding boxes and class labels.
    """
    # Run inference on the image or video
    results = model(image)
    
    # Get the detected bounding boxes and class labels (xywh format: x_center, y_center, width, height, class)
    positions = results.xywh[0].cpu().numpy()  # Extract coordinates
    
    # Filter positions based on bounding box size (width and height should be between 2 and 4)
    filtered_positions = []
    for pos in positions:
        x_center, y_center, width, height, class_id, confidence = pos
        
        if 2 <= width <= 4 and 2 <= height <= 4:  # Filter based on size
            filtered_positions.append(pos)
    
    return filtered_positions

def get_angle(center_x, center_y, point_x, point_y):
    """
    Calculate the angle between the center of the image and a point (e.g., player position).
    
    Args:
    - center_x: X-coordinate of the center of the image.
    - center_y: Y-coordinate of the center of the image.
    - point_x: X-coordinate of the point (player position).
    - point_y: Y-coordinate of the point (player position).
    
    Returns:
    - angle: Angle in degrees between the center and the point.
    """
    delta_x = point_x - center_x
    delta_y = point_y - center_y
    angle = np.arctan2(delta_y, delta_x) * 180 / np.pi
    return angle

def determine_fielder_position(angle):
    """
    Determines the fielder position based on the angle from the center of the image.
    
    Args:
    - angle: Angle in degrees from the center of the image.
    
    Returns:
    - position: Fielding position as a string (e.g., "slip", "cover", etc.).
    """
    if 45 <= angle < 135:
        return "Cover Fielder"
    elif -45 <= angle < 45:
        return "Mid-Off Fielder"
    elif -135 <= angle < -45:
        return "Slip Fielder"
    else:
        return "Leg Slip Fielder"

def draw_boxes(image, positions, front_view=True):
    """
    Draws bounding boxes on the image based on detected positions and determines fielding position.
    
    Args:
    - image: The frame of the video.
    - positions: Detected positions containing bounding boxes and class labels.
    - front_view: Boolean to indicate if the view is from the front or rear.
    """
    height, width, _ = image.shape
    center_x, center_y = width // 2, height // 2  # Center of the image
    
    for pos in positions:
        # Extract coordinates and class label
        x_center, y_center, width, height, class_id, confidence = map(int, pos[:6])
        
        # Get the class label name (e.g., 'ball', 'right-handed', 'left-handed')
        class_name = model_ball.names[class_id] if class_id in model_ball.names else 'Unknown'
        
        # Calculate the angle for fielder position
        player_center_x = int(x_center)
        player_center_y = int(y_center)
        angle = get_angle(center_x, center_y, player_center_x, player_center_y)
        
        # Determine fielder position based on angle
        position = determine_fielder_position(angle)
        
        # Reverse positions if it's from the rear view
        if not front_view:
            angle = -angle
            position = "Reverse " + position
        
        # Draw the bounding box
        x1, y1, x2, y2 = int(x_center - width / 2), int(y_center - height / 2), int(x_center + width / 2), int(y_center + height / 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 4)
        
        # Put the class label and position
        cv2.putText(image, f"{class_name} - {position}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 4)

def process_video(video_path, front_view=True):
    """
    Processes the given video, detects ball and batsman, and draws bounding boxes with fielding positions.
    
    Args:
    - video_path: Path to the video.
    - front_view: Boolean to indicate if the view is from the front or rear.
    """
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get positions from Model 1 (ball detection)
        ball_positions = get_positions(model_ball, frame)
        
        # Get positions from Model 2 (right-handed/left-handed batsman detection)
        batsman_positions = get_positions(model_batsman, frame)
        
        # Draw bounding boxes for ball and batsman detections
        draw_boxes(frame, ball_positions, front_view)
        draw_boxes(frame, batsman_positions, front_view)
        
        # Display the frame with bounding boxes
        cv2.imshow("Video", frame)
        
        # Press 'q' to quit the video window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def select_video():
    """
    Opens a file dialog to select a video from the PC and processes it.
    """
    Tk().withdraw()  # Hide the root window
    video_path = askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
    
    if video_path:
        process_video(video_path, front_view=True)

# Call the select_video function to open the file dialog
select_video()
