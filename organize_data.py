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

    # Define hierarchical classification labels
    disease_labels = {"DR", "ODC", "ARMD", "MH"}
    valid_labels = {"NORMAL"} | disease_labels

    # Ensure top-level folders exist
    normal_dir = os.path.join(output_dir, "NORMAL")
    disease_dir = os.path.join(output_dir, "DISEASE")
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(disease_dir, exist_ok=True)

    # Ensure subfolders for diseases exist
    for label in disease_labels:
        label_dir = os.path.join(disease_dir, label)
        os.makedirs(label_dir, exist_ok=True)

    # Process each image
    for _, row in train_df.iterrows():
        image_name = str(
            row[train_df.columns[0]]
        ).strip()  # Ensure filename is a string and strip spaces
        image_path_png = os.path.join(data_dir, "DatasetA_" + image_name + ".png")

        if (
            "Disease_Risk" in train_df.columns and row["Disease_Risk"] == 0
        ):  # Classify as NORMAL
            dest_path = os.path.join(normal_dir, "DatasetA_" + image_name + ".png")
        else:
            # Classify as DISEASE and further classify into disease categories
            assigned = False
            for label in disease_labels:
                if label in train_df.columns and row[label] == 1:
                    dest_path = os.path.join(
                        disease_dir, label, "DatasetA_" + image_name + ".png"
                    )
                    assigned = True
                    break  # Assign to the first matching disease label
            if not assigned:
                continue

        if os.path.exists(image_path_png):
            shutil.copy(image_path_png, dest_path)

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
