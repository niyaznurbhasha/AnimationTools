import labelme
import numpy as np
import cv2
import os

# File paths
json_path = "Subject3.json"
image_path = os.path.splitext(json_path)[0] + ".png"  # Subject3.png
output_mask = "shirt_mask.png"

# Load image manually
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
H, W = image.shape[:2]

# Load LabelMe JSON
label_file = labelme.LabelFile(filename=json_path)

# Create blank mask
mask = np.zeros((H, W), dtype=np.uint8)

# Combine all polygons labeled "shirt"
for shape in label_file.shapes:
    if shape["label"].strip().lower() == "1":
        points = np.array(shape["points"], dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)

# Feather the edges
mask = cv2.GaussianBlur(mask, (21, 21), 0)

# Save final shirt mask
cv2.imwrite(output_mask, mask)
print(f"✅ Saved binary shirt mask to {output_mask}")
