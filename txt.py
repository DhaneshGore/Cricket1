import tkinter as tk
from tkinter import filedialog
import os

# Initialize Tkinter root window
root = tk.Tk()
root.withdraw()

# Function to allow file selection
def select_files():
    file_paths = filedialog.askopenfilenames(
        title="Select .txt files",
        filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
    )
    return file_paths

# Dictionary to store selected files for each object
selected_files = {}

# Objects list
objects = ["bat", "ball", "umpire", "bowler", "batsman", "fielder", "wicketkeeper", "stumps"]

# Loop through each object
for obj in objects:
    print(f"Select .txt files for {obj}:")
    files = select_files()
    if files:
        selected_files[obj] = files
    else:
        print(f"No files selected for {obj}.")

# All selected .txt files are collected ✅
print("\n✅ All object files selected. Now processing...")

# Path to save final combined file
save_path = filedialog.asksaveasfilename(
    defaultextension=".txt",
    filetypes=[("Text files", "*.txt")],
    title="Save final merged .txt file"
)

# Merging all selected files
with open(save_path, 'w') as outfile:
    for obj, files in selected_files.items():
        for file_path in files:
            with open(file_path, 'r') as infile:
                contents = infile.read()
                outfile.write(contents)
                outfile.write('\n')  # New line after each file content

print(f"\n🎯 Final merged file saved at: {save_path}")
