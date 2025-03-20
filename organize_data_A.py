import os
import shutil
import pandas as pd


def name_images(folder_path, prefix):
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print("Error: Folder does not exist.")
    else:
        for filename in os.listdir(folder_path):
            old_path = os.path.join(folder_path, filename)

            # Check if it's a file (not a directory)
            if os.path.isfile(old_path):
                name, ext = os.path.splitext(filename)
                new_filename = f"{prefix}{name}{ext}"
                new_path = os.path.join(folder_path, new_filename)

                os.rename(old_path, new_path)
    print(
        f"Renaming completed! Files in '{folder_path}' now have the prefix '{prefix}'."
    )


def organize_images(data_type):
    # Define paths
    data_dir = f"data/{data_type}_images"  # Folder containing images
    csv_file = f"data/{data_type}_data_A.csv"  # CSV file containing labels
    output_dir = f"data/{data_type}"  # Destination folder

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load CSV file
    train_df = pd.read_csv(csv_file)
    labels = train_df.columns[1:]  # Extract label names (skip first column)

    # Keep only specified labels
    valid_labels = {"NORMAL", "DR", "ODC", "ARMD", "MH"}
    labels = [label for label in labels if label in valid_labels]

    # Ensure label folders exist
    for label in valid_labels:
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)

    # Process each image
    for _, row in train_df.iterrows():
        image_name = str(
            row[train_df.columns[0]]
        ).strip()  # Ensure filename is a string and strip spaces
        image_path_png = os.path.join(data_dir, "DatasetA_" + image_name + ".png")

        # Check for NORMAL category
        if row[labels[0]] == 0:
            dest_path = os.path.join(
                output_dir, "NORMAL", "DatasetA_" + image_name + ".png"
            )
            if os.path.exists(image_path_png):
                shutil.copy(image_path_png, dest_path)
            else:
                print(f"Warning: {image_path_png} not found.")
        else:
            # Assign image to valid labels only
            for label in labels:
                if row[label] == 1:
                    dest_path = os.path.join(
                        output_dir, label, "DatasetA_" + image_name + ".png"
                    )
                    if os.path.exists(image_path_png):
                        shutil.copy(image_path_png, dest_path)
                    else:
                        print(f"Warning: {image_path_png} not found.")

    print(f"{data_type.capitalize()} image organization complete.")


folder_paths = [
    "data/Train_Images",
    "data/Test_Images",
    "Data/Val_Images",
]
prefix = "DatasetA_"  # Change this for different datasets
"""
for folder_path in folder_paths:
    name_images(folder_path, prefix)
"""

# Run for train, test, and eval
datasets = ["Train", "Test", "Val"]
for dataset in datasets:
    organize_images(dataset)
