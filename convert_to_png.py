from PIL import Image
import os

# Define input and output directories
input_dir = "data/images"  # Folder containing .tif images
output_dir = "data/images"  # Folder to save .png images

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through all .tif files in the input directory
for file_name in os.listdir(input_dir):
    if file_name.lower().endswith(".tif"):
        tif_path = os.path.join(input_dir, file_name)
        png_path = os.path.join(output_dir, file_name.replace(".tif", ".png"))

        # Open the .tif image and convert to .png
        with Image.open(tif_path) as img:
            img.save(png_path, "PNG")
            print(f"Converted: {file_name} -> {png_path}")

        # Delete the original .tif file
        os.remove(tif_path)
        print(f"Deleted: {file_name}")
print("Conversion complete!")
