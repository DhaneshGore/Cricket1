import streamlit as st
import os
import subprocess
import sys
import streamlit.components.v1 as components
import time

# === Page Setup ===
st.set_page_config(layout="wide")
st.title("🏏 Cricket Vision AI – Interactive What-If System")

# === Module Paths ===
MODULES = {
    "1. Object Detection": os.path.join("YOLO", "t1.py"),
    "2. Ball Speed & Trajectory": os.path.join("YOLO", "t22.py"),
    "3. Counterfactual Prediction": os.path.join("YOLO", "t3.py"),
    "4. Bat Swing Analysis": os.path.join("YOLO_2", "b2.py"),
    "5. Shot Style Detection": os.path.join("YOLO_2", "b1.py"),
    "6. Player Positioning & Stance": os.path.join("YOLO_3", "1a.py"),
    "7. Actual vs Hypothetical Shot": os.path.join("YOLO_3", "5.py")
}

# === Select and Run Module ===
option = st.selectbox("🎯 Select a module to run", list(MODULES.keys()))

if st.button("▶️ Run Selected Module"):
    script_path = MODULES[option]
    st.write(f"🔄 Running `{script_path}`...")
    try:
        subprocess.run([sys.executable, script_path], check=True)
        st.success("✅ Module finished successfully.")
        time.sleep(1)  # Give OS time to flush file handles
    except subprocess.CalledProcessError as e:
        st.error(f"❌ Error running `{script_path}`:\n{e}")
    except Exception as ex:
        st.error(f"⚠️ Unexpected error: {ex}")

# === Debug Info ===
st.markdown("---")
st.subheader("🔍 Debug Info")
cwd = os.getcwd()
st.write("📁 Current working directory:", cwd)

assets_dir = os.path.join(cwd, "assets")
if os.path.exists(assets_dir):
    st.write("📂 Files in /assets:", os.listdir(assets_dir))
else:
    st.warning("⚠️ 'assets' folder not found.")

# === Video Output Section ===
st.markdown("---")
st.header("🎬 Video Output Comparison")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📽️ Actual Scenario")
    actual_path = os.path.join("assets", "actual_play.mp4")
    if os.path.exists(actual_path):
        st.write("📦 Size:", os.path.getsize(actual_path), "bytes")
        with open(actual_path, 'rb') as f:
            st.video(f.read())
    else:
        st.warning("⚠️ actual_play.mp4 not found in /assets folder.")

with col2:
    st.subheader("💡 What-If Scenario")
    whatif_path = os.path.join("assets", "whatif_play.mp4")
    if os.path.exists(whatif_path):
        st.write("📦 Size:", os.path.getsize(whatif_path), "bytes")
        with open(whatif_path, 'rb') as f:
            st.video(f.read())
    else:
        st.warning("⚠️ whatif_play.mp4 not found in /assets folder.")

# === WebGL Viewer Integration ===
st.markdown("---")
st.subheader("🧠 Volumetric 3D Viewer (Unity WebGL)")

webgl_url = "http://localhost:5000"
try:
    components.iframe(webgl_url, height=600)
except Exception as e:
    st.warning(f"⚠️ WebGL server not accessible at {webgl_url}. Error: {e}")
