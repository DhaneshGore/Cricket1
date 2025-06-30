# Cricket YOLO Application

This application provides a menu-driven interface to run various cricket analytics tasks using Python scripts. Each task corresponds to a different aspect of cricket video analysis, such as object detection, ball tracking, bat swing analysis, and more.

## Prerequisites

- **Python 3.7+** must be installed on your system.
- (Recommended) Use the provided virtual environment for dependencies.
- All required Python packages should be installed in the environment (see below).

## Setup Instructions

1. **Navigate to the Project Directory**


# === Define full paths to individual task scripts ===
SCRIPT_PATHS = [ 
    "D:/Downloads/Cricket/YOLO/t1.py",         # 1. Object Detection
    "D:/Downloads/Cricket/YOLO/t22.py",        # 2. Ball Tracking & Speed
    "D:/Downloads/Cricket/YOLO/t3.py",         # 3. Counterfactual Ball Prediction
    "D:/Downloads/Cricket/YOLO/b2.py",         # 4. Bat Swing and Angle
    "D:/Downloads/Cricket/Yolo_2/b1.py",       # 5. Shot Execution Classification
    "D:/Downloads/Cricket/Yolo_3/1a.py",       # 6. Player Positioning
    "D:/Downloads/Cricket/Yolo_3/5.py",        # 7. Shot Style Vision


   Open a terminal or command prompt and navigate to the `Cricket` directory:
   ```sh
   cd /d/Downloads/Cricket
   ```

2. **Activate the Virtual Environment**

   If you have a virtual environment set up in `Cricket/YOLO/venv`, activate it:

   - **On Windows (PowerShell):**
     ```sh
     .\venv\Scripts\Activate.ps1
     ```
   - **On Windows (Command Prompt):**
     ```sh
     .\venv\Scripts\activate.bat


## Running the Application

1. **Start the Menu Application**

   In the activated environment, run:
   ```sh
   python central.py
   ```

2. **Using the Menu**

   - You will see a menu with numbered tasks.
   - Enter the number corresponding to the task you want to run and press Enter.
   - To exit, enter `8`.

## Tasks Available

1. Object Detection
2. Ball Tracking & Speed
3. Counterfactual Ball Prediction
4. Bat Swing and Angle
5. Shot Execution Classification
6. Player Positioning
7. Shot Style Vision

## Notes
- Ensure all input files required by the scripts are present in the correct directories.
- Output files will be generated as per each script's logic.
- If you encounter errors, check that all dependencies are installed and paths are correct.

---

For further details, refer to the comments in `central.py` or the individual task scripts. 