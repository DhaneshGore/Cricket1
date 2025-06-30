import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ====== SETTINGS ======
merged_txt_path = 'D:/Downloads/Cricket/y8.txt'  # <-- Your big .txt file with all data
all_images_folders = [
    'D:/Downloads/Cricket/BALL',
    'D:/Downloads/Cricket/BAT',
    'D:/Downloads/Cricket/UMPIRE',
    'D:/Downloads/Cricket/BOWLER',
    'D:/Downloads/Cricket/Batsman',
    'D:/Downloads/Cricket/fielder',
    'D:/Downloads/Cricket/WK',
    'D:/Downloads/Cricket/STUMPS'
]  # <-- Your 8 object folders (images only)

dataset_output_folder = 'D:/Downloads/Cricket/yolo'  # <-- Where final dataset will be saved

train_ratio = 0.8  # 80% training, 20% validation
random_seed = 42
# ========================

# Create output directories
os.makedirs(os.path.join(dataset_output_folder, 'images', 'train'), exist_ok=True)
os.makedirs(os.path.join(dataset_output_folder, 'images', 'val'), exist_ok=True)
os.makedirs(os.path.join(dataset_output_folder, 'labels', 'train'), exist_ok=True)
os.makedirs(os.path.join(dataset_output_folder, 'labels', 'val'), exist_ok=True)

# Load merged .txt file
df = pd.read_csv(merged_txt_path, sep='\s+', header=0)
df.columns = df.columns.str.strip()  # Remove extra spaces

# Group by images
grouped = df.groupby('Image_Name')

image_names = list(grouped.groups.keys())

# Train-val split
train_imgs, val_imgs = train_test_split(image_names, train_size=train_ratio, random_state=random_seed)

def copy_images_and_labels(img_list, split):
    for img_name in tqdm(img_list, desc=f'Copying {split} data'):
        # Find image file from one of the folders
        found = False
        for folder in all_images_folders:
            img_path = os.path.join(folder, img_name)
            if os.path.exists(img_path):
                found = True
                break
        if not found:
            print(f"❌ Image {img_name} not found in given folders.")
            continue
        
        # Copy image
        shutil.copy(img_path, os.path.join(dataset_output_folder, 'images', split, img_name))
        
        # Create label .txt
        label_lines = []
        for idx, row in grouped.get_group(img_name).iterrows():
            label_lines.append(f"{row['class_id']} {row['x_center']} {row['y_center']} {row['width']} {row['height']}")
        
        label_txt_path = os.path.join(dataset_output_folder, 'labels', split, img_name.replace('.png', '.txt').replace('.jpg', '.txt'))
        with open(label_txt_path, 'w') as f:
            f.write('\n'.join(label_lines))

# Copy images and labels
copy_images_and_labels(train_imgs, 'train')
copy_images_and_labels(val_imgs, 'val')

# Create data.yaml file (for YOLOv8 training)
yaml_content = f"""
path: {dataset_output_folder}
train: images/train
val: images/val

nc: 8
names: ['ball', 'bat', 'umpire', 'bowler', 'batsman', 'fielder', 'wicketkeeper', 'stumps']
"""

with open(os.path.join(dataset_output_folder, 'data.yaml'), 'w') as f:
    f.write(yaml_content)

print("\n✅ Dataset prepared successfully!")
