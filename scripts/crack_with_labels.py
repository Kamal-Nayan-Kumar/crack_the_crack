import os
import cv2
import numpy as np
from detect_bottles import detect_bottles

# Paths
input_folder = "tobecracked"
output_image_folder = "dataset_cracked/train/images"
output_label_folder = "dataset_cracked/train/labels"
label_source_folder = "dataset/crack_labels"  # Original labels folder

os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_label_folder, exist_ok=True)

def generate_crack_pattern(image, bbox):
    """
    Generates a realistic thin crack inside the detected bottle region.
    Returns the modified image and the crack bounding box.
    """
    x1, y1, x2, y2 = map(int, bbox)

    # Ensure the bounding box is large enough for crack generation
    if (x2 - x1) < 50 or (y2 - y1) < 50:
        return image, None  # Skip small bottles

    overlay = image.copy()

    # Define crack starting point inside the bounding box
    start_x = np.random.randint(x1 + 10, x2 - 10)
    start_y = np.random.randint(y1 + 10, y2 - 10)

    # Generate a curved crack shape
    crack_points = [(start_x, start_y)]
    num_points = np.random.randint(8, 12)

    for _ in range(num_points):
        next_x = np.clip(crack_points[-1][0] + np.random.randint(-15, 15), x1, x2)
        next_y = np.clip(crack_points[-1][1] + np.random.randint(10, 20), y1, y2)
        crack_points.append((next_x, next_y))

    # Draw the main crack line
    for i in range(len(crack_points) - 1):
        cv2.line(overlay, crack_points[i], crack_points[i + 1], (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    # Compute crack bounding box
    crack_x_min = min(p[0] for p in crack_points)
    crack_y_min = min(p[1] for p in crack_points)
    crack_x_max = max(p[0] for p in crack_points)
    crack_y_max = max(p[1] for p in crack_points)

    crack_bbox = (crack_x_min, crack_y_min, crack_x_max, crack_y_max)

    return cv2.addWeighted(overlay, 0.8, image, 0.2, 0), crack_bbox

def convert_bbox_to_yolo(image_shape, bbox):
    """
    Converts a bounding box (x1, y1, x2, y2) to YOLO format.
    """
    img_h, img_w = image_shape[:2]
    x1, y1, x2, y2 = bbox

    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h

    return f"1 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"  # Class 1 for crack

def generate_synthetic_data(image_path, label_path, output_image_path, output_label_path):
    """
    Detects bottles, adds cracks, and updates the YOLO label file.
    """
    image, boxes = detect_bottles(image_path)
    new_labels = []

    # Read existing bottle labels
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            new_labels = f.readlines()

    # Process each detected bottle
    for bbox in boxes:
        image, crack_bbox = generate_crack_pattern(image, bbox)

        if crack_bbox:
            # Convert crack bounding box to YOLO format
            yolo_crack_label = convert_bbox_to_yolo(image.shape, crack_bbox)
            new_labels.append(yolo_crack_label)

    # Save the cracked image
    cv2.imwrite(output_image_path, image)

    # Save the updated label file
    with open(output_label_path, "w") as f:
        f.writelines(new_labels)

# Process all images in the dataset
for image_name in os.listdir(input_folder):
    if image_name.endswith(".jpg") or image_name.endswith(".png"):
        image_path = os.path.join(input_folder, image_name)
        label_path = os.path.join(label_source_folder, image_name.replace(".jpg", ".txt").replace(".png", ".txt"))

        output_image_path = os.path.join(output_image_folder, image_name)
        output_label_path = os.path.join(output_label_folder, image_name.replace(".jpg", ".txt").replace(".png", ".txt"))

        generate_synthetic_data(image_path, label_path, output_image_path, output_label_path)

print("✅ Synthetic crack dataset with updated labels created successfully!")
