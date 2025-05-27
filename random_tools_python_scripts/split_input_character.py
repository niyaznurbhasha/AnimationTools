from pathlib import Path
import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
from tf_bodypix.draw import draw_poses
import numpy as np
from PIL import Image
import os

def calculate_centroid(mask):
    """Calculate the centroid of a binary mask."""
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None
    centroid_x = np.mean(x_indices)
    centroid_y = np.mean(y_indices)
    return centroid_x, centroid_y


# Setup input and output paths
output_path = Path('./segmented_input')
output_path.mkdir(parents=True, exist_ok=True)

local_input_path = '/mnt/c/Users/niyaz/Downloads/man.jpg'  # Update this to the path of your local image

# Load the BodyPix model (only needs to be done once)
bodypix_model = load_model(download_model(BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16))

# Get prediction results
image = tf.keras.preprocessing.image.load_img(local_input_path)
image_array = tf.keras.preprocessing.image.img_to_array(image)
result = bodypix_model.predict_single(image_array)

# Generate a simple binary mask
mask = result.get_mask(threshold=0.75).numpy()  # Convert EagerTensor to NumPy array

# Ensure mask is in the correct format (convert to grayscale if necessary)
if mask.ndim == 2:  # Single-channel mask
    mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
elif mask.shape[2] == 1:  # Handle (H, W, 1) shape
    mask_image = Image.fromarray(mask[:, :, 0] * 255, mode='L')
else:
    raise ValueError("Unexpected mask shape.")

mask_image.save(output_path / 'output-mask.png')

# Generate a colored mask for different body parts
colored_mask = result.get_colored_part_mask(mask)  # No need to call .numpy() here

# Ensure colored_mask is in the correct format
if colored_mask.ndim == 3 and colored_mask.shape[2] == 3:  # RGB colored mask
    colored_mask_image = Image.fromarray(colored_mask.astype(np.uint8), mode='RGB')
else:
    raise ValueError("Unexpected colored_mask shape.")

colored_mask_image.save(output_path / 'output-colored-mask.png')

# Create directory for segmented body parts
segmented_output_path = output_path / "segmented_parts"
segmented_output_path.mkdir(parents=True, exist_ok=True)

# Identify unique colors in the colored mask
unique_colors = np.unique(colored_mask.reshape(-1, colored_mask.shape[2]), axis=0)

def extract_part_from_colored_mask(image, colored_mask, color):
    """Extracts the part of the image corresponding to a specific color in the colored mask."""
    part_mask = np.all(colored_mask == color, axis=-1)
    part_image = np.zeros_like(image)
    part_image[part_mask] = image[part_mask]
    return part_image, part_mask

centroids = {}

# Process each unique color and save the corresponding part image
for i, color in enumerate(unique_colors):
    part_image, part_mask = extract_part_from_colored_mask(image_array, colored_mask, color)
    part_image_pil = Image.fromarray(part_image.astype(np.uint8))
    part_image_pil.save(segmented_output_path / f"part_{i}.png")

    # Calculate and store the centroid
    centroid = calculate_centroid(part_mask)
    if centroid:
        centroids[f'part_{i}'] = centroid

# Print the centroids
print("Centroids of the segmented body parts:")
for part_name, centroid in centroids.items():
    print(f"{part_name}: {centroid}")

# Draw poses on the image and save the result
poses = result.get_poses()
image_with_poses = draw_poses(image_array.copy(), poses, keypoints_color=(255, 100, 100), skeleton_color=(100, 100, 255))
image_with_poses_image = Image.fromarray(image_with_poses.astype(np.uint8))
image_with_poses_image.save(f'{output_path}/output-poses.jpg')
