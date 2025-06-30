import subprocess
import sys
import os

# === Define full paths to individual task scripts ===
SCRIPT_PATHS = [ 
    "D:/Downloads/Cricket/YOLO/t1.py",         # 1. Object Detection
    "D:/Downloads/Cricket/YOLO/t22.py",        # 2. Ball Tracking & Speed
    "D:/Downloads/Cricket/YOLO/t3.py",         # 3. Counterfactual Ball Prediction
    "D:/Downloads/Cricket/YOLO/b2.py",         # 4. Bat Swing and Angle
    "D:/Downloads/Cricket/Yolo_2/b1.py",       # 5. Shot Execution Classification
    "D:/Downloads/Cricket/Yolo_3/1a.py",       # 6. Player Positioning
    "D:/Downloads/Cricket/Yolo_3/5.py",        # 7. Shot Style Vision
]

# === Function to run selected script ===
def run_script(script_path):
    try:
        print(f"🚀 Running: {os.path.basename(script_path)}")
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Script failed with error: {e}")
    except Exception as e:
        print(f"⚠️ Unexpected error while running script: {e}")

# === Display menu to user ===
def display_menu():
    print("\n📋 Select a task to run:")
    print("1. Object Detection")
    print("2. Ball Tracking & Speed")
    print("3. Counterfactual Ball Prediction")
    print("4. Bat Swing and Angle")
    print("5. Shot Execution Classification")
    print("6. Player Positioning")
    print("7. Shot Style Vision")
    print("8. Exit")

# === Main control loop ===
def main():
    while True:
        display_menu()
        try:
            choice = int(input("Enter the number of your choice: "))
            if 1 <= choice <= len(SCRIPT_PATHS):
                run_script(SCRIPT_PATHS[choice - 1])
            elif choice == 8:
                print("👋 Exiting the menu. Goodbye!")
                break
            else:
                print("⚠️ Invalid choice. Please choose a valid number.")
        except ValueError:
            print("❗ Please enter a number.")

if __name__ == "__main__":
    main()
