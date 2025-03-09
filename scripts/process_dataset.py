import os
from generate_cracks import generate_synthetic_data

input_folder = "tobecracked"
output_folder = "dataset_cracked/train/images"

os.makedirs(output_folder, exist_ok=True)

for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)
    output_path = os.path.join(output_folder, image_name)
    generate_synthetic_data(image_path, output_path)

print("Synthetic crack dataset created successfully!")