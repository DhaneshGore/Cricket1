import cv2

video_path = 'D:\\Downloads\\6.mp4'  # Update this path if your file is elsewhere

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)

print(f"🎯 FPS of the video: {fps}")

cap.release()
