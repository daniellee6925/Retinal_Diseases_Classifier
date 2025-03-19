import os

folder_path = "data/Val_images"  # Change this to your dataset folder
prefix = "DatasetA_"  # Change this for different datasets

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
