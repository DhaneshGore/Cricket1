import subprocess
import sys
import os
from tkinter import Tk, filedialog

# === Paths to all individual task scripts ===
SCRIPT_PATHS = [
    "D:/Downloads/Cricket/YOLO/t1.py",         # Object Detection
    "D:/Downloads/Cricket/YOLO/t22.py",        # Ball Tracking & Speed
    "D:/Downloads/Cricket/YOLO/t3.py",         # Counterfactual Ball Prediction
    "D:/Downloads/Cricket/YOLO/b2.py",         # Bat Swing and Angle
    "D:/Downloads/Cricket/YOLO_2/b1.py",       # Shot Execution Classification
    "D:/Downloads/Cricket/Yolo_3/1a.py",       # Player Positioning
    "D:/Downloads/Cricket/Yolo_3/5.py"         # Shot Style Vision
]

# === Function to run a script with the selected video path ===
def run_script(script_path, video_path):
    try:
        print(f"\n🚀 Running {os.path.basename(script_path)}...")
        subprocess.run([sys.executable, script_path, video_path], check=True)
        print(f"✅ {os.path.basename(script_path)} completed.\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ Script {script_path} failed with error: {e}")
    except Exception as e:
        print(f"⚠️ Unexpected error running {script_path}: {e}")

def main():
    print("📂 Please select your cricket video...")
    Tk().withdraw()
    video_path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
    )

    if not video_path:
        print("❌ No video selected. Exiting.")
        return

    print(f"🎞️ Selected video: {video_path}")

    # Run all modules on the selected video
    for script_path in SCRIPT_PATHS:
        run_script(script_path, video_path)

    print("🎉 All modules completed successfully!")

if __name__ == "__main__":
    main()
