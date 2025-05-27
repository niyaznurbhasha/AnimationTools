import cv2
import numpy as np
import os
import glob

def apply_offset_to_contour(contour, offset):
    # Add offset to each point in the contour
    return contour + offset

def find_and_draw_contours(image_path, output_path, threshold_value=127, offset=(0, 0)):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}. Check the file path and file integrity.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create a black background image
    black_background = np.zeros_like(image)

    # Apply offset and draw contours on the black background
    for contour in contours:
        offset_contour = apply_offset_to_contour(contour, np.array(offset))
        cv2.drawContours(black_background, [offset_contour], -1, (0, 255, 0), 3)

    # Save the result
    cv2.imwrite(output_path, black_background)


def process_images(image_folder, threshold_value=127):
    # Get all image paths
    image_paths = glob.glob(os.path.join(image_folder, '*'))

    # Create output directory
    output_folder = os.path.join(image_folder, 'output')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image
    for image_path in image_paths:
        base_name = os.path.basename(image_path)
        output_path = os.path.join(output_folder, base_name)
        find_and_draw_contours(image_path, output_path, threshold_value)

process_images('path/to/your/image/directory', offset=(10, 15))

