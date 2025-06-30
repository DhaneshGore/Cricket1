import tkinter as tk
from tkinter import filedialog
import os

# Initialize Tkinter root window
root = tk.Tk()
root.withdraw()

def select_files():
    file_paths = filedialog.askopenfilenames(
        title="Select .txt files",
        filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
    )
    return file_paths

selected_files = {}
objects = ["ball", "bat", "umpire", "bowler", "batsman", "fielder", "wicketkeeper", "stumps"]

for obj in objects:
    print(f"Select .txt files for {obj}:")
    files = select_files()
    if files:
        selected_files[obj] = files
    else:
        print(f"No files selected for {obj}.")

print("\nAll object files selected. Now processing...")

save_path = filedialog.asksaveasfilename(
    defaultextension=".txt",
    filetypes=[("Text files", "*.txt")],
    title="Save final merged .txt file"
)

if not save_path:
    print("No save path selected. Exiting.")
    exit()

with open(save_path, 'w') as outfile:
    for obj, files in selected_files.items():
        for file_path in files:
            with open(file_path, 'r') as infile:
                contents = infile.read()
                outfile.write(contents)
                outfile.write('\n')  # New line after each file content

print(f"\nFinal merged file saved at: {save_path}")
