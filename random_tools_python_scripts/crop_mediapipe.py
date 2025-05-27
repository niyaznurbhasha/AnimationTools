import cv2
import numpy as np
import mediapipe as mp
import json
import os

def extract_keypoints(image_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        raise ValueError("No keypoints detected")

    keypoints = []
    for landmark in results.pose_landmarks.landmark:
        keypoints.append((int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])))

    # Neck point approximated as the midpoint between shoulders
    left_shoulder = keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    neck_point = ((left_shoulder[0] + right_shoulder[0]) // 2,
                  (left_shoulder[1] + right_shoulder[1]) // 2)

    return keypoints, neck_point

def crop_image_at_neck(image_path, neck_point, output_upper, output_lower):
    # Load the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    
    # Crop the image into upper and lower parts based on the neck point
    upper_image = image[:neck_point[1], :]
    lower_image = image[neck_point[1]:, :]

    # Save the cropped images
    cv2.imwrite(output_upper, upper_image)
    cv2.imwrite(output_lower, lower_image)

# Paths
image_path = "/Users/niyaz/Downloads/alexander.png"
output_json_dir = "/Users/niyaz/Downloads/alexander_mediapipe"
output_json_file = os.path.join(output_json_dir, "keypoints.json")
output_upper = "/Users/niyaz/Downloads/alexander_mediapipe/output_upper.png"
output_lower = "/Users/niyaz/Downloads/alexander_mediapipe/output_lower.png"

# Ensure the output directory exists
os.makedirs(output_json_dir, exist_ok=True)

# Extract keypoints using MediaPipe
keypoints, neck_point = extract_keypoints(image_path)

# Crop the image based on the neck point and save the outputs
crop_image_at_neck(image_path, neck_point, output_upper, output_lower)

# Save the keypoints to a JSON file
with open(output_json_file, 'w') as f:
    json.dump({"keypoints": keypoints}, f)

print(f"Keypoints saved to {output_json_file}")
print(f"Upper body image saved to {output_upper}")
print(f"Lower body image saved to {output_lower}")
