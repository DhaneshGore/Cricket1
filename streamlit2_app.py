# streamlit_app.py

import streamlit as st
import os
import shutil
import sys
import tempfile
import subprocess

# === Page Settings ===
st.set_page_config(layout="wide")
st.title("🏏 Cricket Vision AI – All-in-One Video Analyzer")

# === Upload Video ===
uploaded_file = st.file_uploader("📁 Upload a Cricket Video (.mp4 or .mov)", type=["mp4", "mov"])

if uploaded_file:
    # Save uploaded file to a temporary location
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, uploaded_file.name)

    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ Video uploaded: {uploaded_file.name}")

    # === Run All Modules Sequentially ===
    st.info("Running all modules on uploaded video...")

    module_paths = [
        "YOLO/t1.py",        # Object Detection
        "YOLO/t22.py",       # Ball Speed & Tracking
        "YOLO/t3.py",        # Counterfactual Prediction
        "YOLO/b2.py",        # Bat Swing & Angle
        "YOLO_2/b1.py",      # Shot Execution
        "Yolo_3/1a.py",      # Player Positioning
        "Yolo_3/5.py",       # Shot Style Vision
    ]

    for script in module_paths:
        st.write(f"▶️ Running `{script}`...")
        try:
            subprocess.run([sys.executable, script, video_path], check=True)
        except subprocess.CalledProcessError as e:
            st.error(f"❌ Error running {script}: {e}")

    st.success("🎉 All modules completed!")

    # === Show Outputs ===
    st.header("🎬 Results")

    output_folder = os.path.join(os.path.expanduser("~"), "Desktop", "Vraj_Assignment")

    for filename in os.listdir(output_folder):
        if uploaded_file.name.split('.')[0] in filename and filename.endswith(".mp4"):
            st.video(os.path.join(output_folder, filename))

# Optional: Instructions
else:
    st.warning("👆 Please upload a video to get started.")
