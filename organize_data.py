import os
import pandas as pd
import shutil

# Define paths
data_dir = "data/images"  # Folder containing images
train_csv = "data/train_data.csv"  # CSV file containing labels
output_dir = "data/train"  # Destination folder

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load CSV file
train_df = pd.read_csv(train_csv)
labels = train_df.columns[1:]  # Extract label names (skip first column)

# Ensure label folders exist
for label in labels:
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)

# Process each image
for _, row in train_df.iterrows():
    image_name = row[train_df.columns[0]]  # First column contains file name
    image_name = str(
        row[train_df.columns[0]]
    ).strip()  # Ensure filename is a string and strip spaces
    # Check for possible extensions
    image_path_png = os.path.join(data_dir, image_name + ".png")

    # Find the label(s) assigned to the image
    for label in labels:
        if row[label] == 1:  # If one-hot encoding is 1, move the image
            dest_path = os.path.join(output_dir, label, image_name + ".png")
            if os.path.exists(image_path_png):
                shutil.copy(image_path_png, dest_path)
            else:
                print(f"Warning: {image_path_png} not found.")

print("Image organization complete.")
