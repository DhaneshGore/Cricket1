import streamlit as st
import os
import subprocess

st.set_page_config(layout="wide")
st.title("🏏 Cricket Vision AI - Modular Demo")

video_file = st.file_uploader("📤 Upload a cricket video", type=["mp4", "mov", "avi"])
if video_file:
    with open("input_video.mp4", "wb") as f:
        f.write(video_file.read())
    st.success("✅ Video uploaded!")
    st.video("input_video.mp4")

    task = st.selectbox("📂 Choose analysis module", [
        "Object Detection",
        "Ball Tracking & Speed",
        "Counterfactual Ball Prediction",
        "Bat Swing and Angle",
        "Shot Execution Classification",
        "Player Positioning",
        "Shot Style Vision"
    ])

    task_to_script = {
        "Object Detection": "YOLO/t1.py",
        "Ball Tracking & Speed": "YOLO/t22.py",
        "Counterfactual Ball Prediction": "YOLO/t3.py",
        "Bat Swing and Angle": "YOLO/b2.py",
        "Shot Execution Classification": "Yolo_2/b1.py",
        "Player Positioning": "Yolo_3/1a.py",
        "Shot Style Vision": "Yolo_3/5.py",
    }

    if st.button("🚀 Run"):
        script_path = task_to_script[task]
        st.info(f"Running {task}...")

        result = subprocess.run(["python", script_path, "input_video.mp4"])
        st.success("✅ Done!")

        # Try showing outputs
        if os.path.exists("assets/actual_play.mp4"):
            st.video("assets/actual_play.mp4")
        elif os.path.exists("assets/whatif_play.mp4"):
            st.video("assets/whatif_play.mp4")
        else:
            st.warning("⚠️ No output video found.")
