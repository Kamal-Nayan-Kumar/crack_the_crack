import os
import json
import cv2

input_dir = "dataset/labels_json/"   # Folder with LabelMe JSON files
output_dir = "dataset/labels_yolo/"  # Folder for YOLO annotations
image_dir = "dataset/extract_frames/" # Folder with images

os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.endswith(".json"):
        json_path = os.path.join(input_dir, file)
        with open(json_path) as f:
            data = json.load(f)

        # Ensure correct image path
        image_path = os.path.join(image_dir, data["imagePath"])

        # Check if the image exists
        if not os.path.exists(image_path):
            print(f"❌ Warning: Image {image_path} not found. Skipping...")
            continue

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: Could not read {image_path}. Skipping...")
            continue

        height, width, _ = image.shape

        yolo_labels = []
        for shape in data["shapes"]:
            label = 0  # Class ID for "bottle"
            points = shape["points"]
            x_min, y_min = min(points, key=lambda p: p[0])[0], min(points, key=lambda p: p[1])[1]
            x_max, y_max = max(points, key=lambda p: p[0])[0], max(points, key=lambda p: p[1])[1]

            x_center = (x_min + x_max) / 2 / width
            y_center = (y_min + y_max) / 2 / height
            w = (x_max - x_min) / width
            h = (y_max - y_min) / height

            yolo_labels.append(f"{label} {x_center} {y_center} {w} {h}")

        # Save YOLO format annotation
        output_txt = os.path.join(output_dir, file.replace(".json", ".txt"))
        with open(output_txt, "w") as out_f:
            out_f.write("\n".join(yolo_labels))

print("✅ Annotations converted to YOLO format.")
