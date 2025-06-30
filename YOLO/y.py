import os
import shutil
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==== SETTINGS ====
object_folders = {
    'bat': 'D:/Downloads/Cricket/BAT',
    'ball': 'D:/Downloads/Cricket/BALL',
    'umpire': 'D:/Downloads/Cricket/UMPIRE',
    'bowler': 'D:/Downloads/Cricket/BOWLER',
    'batsman': 'D:/Downloads/Cricket/Batsman',
    'fielder': 'D:/Downloads/Cricket/fielder',
    'wicketkeeper': 'D:/Downloads/Cricket/WK',
    'stumps': 'D:/Downloads/Cricket/STUMPS',
}

merged_labels_path = 'D:/Downloads/Cricket/y8.txt'  # tab-separated
output_dataset = 'D:/Downloads/Cricket/YOLO_dataset'
images_output = os.path.join(output_dataset, 'images')
labels_output = os.path.join(output_dataset, 'labels')

train_ratio = 0.8
images_per_object = 16  # 128 total images (8 classes * 16)

# === Create folders ===
for split in ['train', 'val']:
    os.makedirs(os.path.join(images_output, split), exist_ok=True)
    os.makedirs(os.path.join(labels_output, split), exist_ok=True)

# === Load merged label file ===
df = pd.read_csv(merged_labels_path, sep='\t', header=None)
df.columns = ['Image_Name', 'class_id', 'x_center', 'y_center', 'width', 'height']

# === Copy and rename images ===
new_image_label_map = []

for obj_name, obj_folder in object_folders.items():
    all_images = [img for img in os.listdir(obj_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    selected_images = random.sample(all_images, images_per_object)

    for idx, img_name in enumerate(selected_images):
        ext = os.path.splitext(img_name)[1]
        new_img_name = f"{obj_name}_{idx+1}{ext}"
        src_img_path = os.path.join(obj_folder, img_name)
        new_img_path = os.path.join(images_output, new_img_name)

        shutil.copy(src_img_path, new_img_path)
        new_image_label_map.append((img_name, new_img_name))

# === Generate YOLO-format .txt labels ===
for old_name, new_name in tqdm(new_image_label_map, desc="Generating labels"):
    label_rows = df[df['Image_Name'] == old_name]
    if label_rows.empty:
        print(f"⚠️ Warning: No labels found for {old_name}")
        continue

    label_lines = [
        f"{int(row['class_id'])} {row['x_center']} {row['y_center']} {row['width']} {row['height']}"
        for _, row in label_rows.iterrows()
    ]

    new_txt_name = os.path.splitext(new_name)[0] + '.txt'
    new_txt_path = os.path.join(labels_output, new_txt_name)
    with open(new_txt_path, 'w') as f:
        f.write('\n'.join(label_lines))

# === Split into train/val ===
all_images = [img for img in os.listdir(images_output) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
train_imgs, val_imgs = train_test_split(all_images, train_size=train_ratio, random_state=42)

for split, img_list in zip(['train', 'val'], [train_imgs, val_imgs]):
    for img_name in img_list:
        label_name = os.path.splitext(img_name)[0] + '.txt'
        shutil.copy(os.path.join(images_output, img_name), os.path.join(images_output, split, img_name))
        shutil.copy(os.path.join(labels_output, label_name), os.path.join(labels_output, split, label_name))

print("✅ Balanced dataset created successfully!")

# === Write dataset.yaml ===
yaml_content = f"""
path: {output_dataset}
train: images/train
val: images/val
nc: 8
names: ['bat', 'ball', 'umpire', 'bowler', 'batsman', 'fielder', 'wicketkeeper', 'stumps']
"""

with open(os.path.join(output_dataset, 'dataset.yaml'), 'w') as f:
    f.write(yaml_content.strip())

print("✅ dataset.yaml file created successfully!")
