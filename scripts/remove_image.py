import os

# Define folder paths
image_folder = "dataset/extract_frames"
json_folder = "dataset/labels_json"

# Get the set of JSON filenames (without extension)
json_files = {os.path.splitext(f)[0] for f in os.listdir(json_folder) if f.endswith(".json")}

# Iterate through images and remove those without a matching JSON file
for image_file in os.listdir(image_folder):
    if image_file.endswith(".jpg"):
        image_name = os.path.splitext(image_file)[0]  # Remove extension
        if image_name not in json_files:
            image_path = os.path.join(image_folder, image_file)
            os.remove(image_path)
            print(f"Removed: {image_path}")

print("Cleanup complete.")
