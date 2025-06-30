import os
import shutil
import random

# === CONFIG ===
source_dir = "D:/Downloads/Cricket/Yolo_3/all"
output_dir = "D:/Downloads/Cricket/Yolo_3/test"
train_ratio = 0.7

# === SETUP DESTINATION FOLDERS ===
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# === COLLECT IMAGE FILES ===
image_exts = (".jpg", ".jpeg", ".png")
image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(image_exts)]

# === SHUFFLE & SPLIT ===
random.shuffle(image_files)
split_index = int(train_ratio * len(image_files))
train_images = image_files[:split_index]
val_images = image_files[split_index:]

def copy_files(image_list, dest_folder):
    for img_name in image_list:
        label_name = os.path.splitext(img_name)[0] + ".txt"
        shutil.copy(os.path.join(source_dir, img_name), os.path.join(dest_folder, img_name))
        if os.path.exists(os.path.join(source_dir, label_name)):
            shutil.copy(os.path.join(source_dir, label_name), os.path.join(dest_folder, label_name))
        else:
            print(f"⚠️ Warning: No label found for {img_name}")

copy_files(train_images, train_dir)
copy_files(val_images, val_dir)

print(f"✅ Train/Val split done: {len(train_images)} train, {len(val_images)} val")
