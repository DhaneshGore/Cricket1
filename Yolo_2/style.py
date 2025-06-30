import os
import shutil
import random
import yaml

# === 🔧 Modify this path ===
base_dir = "/Users/vrajalpeshkumarmodi/Downloads/Cricket/Yolo_2/st"
# ===========================
split_ratio = 0.7  # 80% train, 20% val
class_names = ['cover_drive', 'defencive', 'Aggresive', 'leave']
image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

# === Setup output folders ===
image_dir = os.path.join(base_dir, "images")
label_dir = os.path.join(base_dir, "labels")
for split in ["train", "val"]:
    os.makedirs(os.path.join(image_dir, split), exist_ok=True)
    os.makedirs(os.path.join(label_dir, split), exist_ok=True)

# === Find all image files and generate .txt from bounding_boxes.txt ===
bbox_file = os.path.join(base_dir, "st.txt")
annotations = {}

with open(bbox_file, 'r') as f:
    lines = f.readlines()[1:]  # Skip header
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) == 6:
            img_name, cls, xc, yc, w, h = parts
            yolo_line = f"{cls} {xc} {yc} {w} {h}\n"
            txt_name = os.path.splitext(img_name)[0] + ".txt"
            annotations.setdefault(txt_name, []).append(yolo_line)

for txt_name, yolo_lines in annotations.items():
    with open(os.path.join(base_dir, txt_name), 'w') as f:
        f.writelines(yolo_lines)

# === Continue with image/txt pairing ===
image_files = [f for f in os.listdir(base_dir) if f.lower().endswith(image_exts)]
image_files = [f for f in image_files if os.path.isfile(os.path.join(base_dir, f))]
random.shuffle(image_files)
split_idx = int(len(image_files) * split_ratio)
train_images = image_files[:split_idx]
val_images = image_files[split_idx:]

# === Move files to train/val folders ===
def move_pair(file_list, split):
    for img_file in file_list:
        name_no_ext = os.path.splitext(img_file)[0]
        txt_file = name_no_ext + ".txt"

        img_src = os.path.join(base_dir, img_file)
        txt_src = os.path.join(base_dir, txt_file)

        img_dst = os.path.join(image_dir, split, img_file)
        txt_dst = os.path.join(label_dir, split, txt_file)

        if os.path.exists(img_src) and os.path.exists(txt_src):
            shutil.copy(img_src, img_dst)
            shutil.copy(txt_src, txt_dst)
        else:
            print(f"⚠️ Skipping {img_file} - missing corresponding .txt file")

move_pair(train_images, "train")
move_pair(val_images, "val")

# === Create data.yaml ===
yaml_dict = {
    "train": os.path.join(base_dir, "images/train"),
    "val": os.path.join(base_dir, "images/val"),
    "nc": len(class_names),
    "names": class_names
}

yaml_path = os.path.join(base_dir, "data.yaml")
with open(yaml_path, 'w') as f:
    yaml.dump(yaml_dict, f)

print(f"\n✅ Done! YOLO dataset prepared at: {base_dir}")
print(f"  - Images and labels split into train/val")
print(f"  - data.yaml created: {yaml_path}")