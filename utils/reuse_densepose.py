import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

import cv2
import numpy as np
from collections import Counter
import cv2
import numpy as np

import cv2
import numpy as np

def process_frame(frame, target_color, tolerance):
    # Split the frame into its RGB components
    R, G, B = cv2.split(frame)
    
    # Check if each component is within the tolerance of the target color's respective component
    R_mask = (R >= target_color[0] - tolerance) & (R <= target_color[0] + tolerance)
    G_mask = (G >= target_color[1] - tolerance) & (G <= target_color[1] + tolerance)
    B_mask = (B >= target_color[2] - tolerance) & (B <= target_color[2] + tolerance)
    
    # Combine the masks to find where all three conditions are met
    mask = R_mask & G_mask & B_mask
    
    # Create a new frame, setting the matched pixels to white and others to black
    new_frame = np.zeros_like(frame)
    new_frame[mask] = [255, 255, 255]  # Set matched pixels to white
    
    return new_frame

def process_video(video_path, output_path, target_color=(255, 255, 0), tolerance=20):
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create a VideoWriter object to write the processed video
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame
        processed_frame = process_frame(frame, target_color, tolerance)
        
        # Write the processed frame to the output video
        out.write(processed_frame)
    
    # Release everything when the job is finished
    cap.release()
    out.release()

# Specify the path to your video and the output video file name
video_path = '/Users/niyaz/Downloads/test_densepose.mp4'  # Update this with the path to your input video
output_path = '/Users/niyaz/Downloads/test_densepose_binary.mp4'  # The output video file

# Process the video
process_video(video_path, output_path)

exit()

def process_frame(frame, target_color, tolerance):
    # Calculate the lower and upper bounds for each color component
    lower_bound = np.array(target_color) - tolerance
    upper_bound = np.array(target_color) + tolerance
    
    # Create a mask where each pixel within the range is marked as True
    mask = np.all((frame >= lower_bound) & (frame <= upper_bound), axis=-1)
    
    # Initialize a new frame with all pixels set to black
    new_frame = np.zeros_like(frame)
    
    # Set the pixels within the range to white
    new_frame[mask] = [255, 255, 255]
    
    return new_frame

def process_video(video_path, output_path, target_color=(140, 115, 43), tolerance=5):
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create a VideoWriter object to write the processed video
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame
        processed_frame = process_frame(frame, target_color, tolerance)
        
        # Write the processed frame to the output video
        out.write(processed_frame)
    
    # Release everything when the job is finished
    cap.release()
    out.release()

# Specify the path to your video and the output video file name
video_path = '/Users/niyaz/Downloads/test_densepose.mp4'  # Update this with the path to your input video
output_path = '/Users/niyaz/Downloads/test_densepose_binary.mp4'  # The output video file

# Process the video
process_video(video_path, output_path)

exit()
def print_color_counts(frame):
    # Reshape the frame to a 2D array where each row is a pixel
    pixels = frame.reshape((-1, 3))
    # Convert pixels to tuples to make them hashable for Counter
    pixel_tuples = [tuple(pixel) for pixel in pixels]
    # Count occurrences of each color
    color_counts = Counter(pixel_tuples)
    # Print color counts
    for color, count in color_counts.items():
        print(f"Color: {color}, Count: {count}")

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        print(f"Frame {frame_count}: Color counts")
        print_color_counts(frame)
        frame_count += 1
        exit()

    cap.release()
    print("Video processing completed.")

video_path = "/Users/niyaz/Downloads/test_densepose.mp4"  # Update this path to your video file
process_video(video_path)
exit()

def get_dominant_colors(frame, k=10):
    # Reshape the frame for KMeans and reduce the number of pixels for faster processing
    data = np.reshape(frame, (-1, 3))
    data = data[::10]  # Reduce the dataset size for faster processing

    # Apply KMeans to find clusters (dominant colors)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    colors = kmeans.cluster_centers_

    # Count the occurrences of each label to find the most common
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    most_common = labels[np.argsort(-counts)][:k]

    return colors[most_common].astype(int), counts[most_common]

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get dominant colors for the current frame
        dominant_colors, counts = get_dominant_colors(frame, k=10)
        print(f"Frame {frame_count}: Top 10 colors (RGB) and their counts")
        for color, count in zip(dominant_colors, counts): 
            print(f"Color: {color}, Count: {count}")

        frame_count += 1

    cap.release()
    print("Video processing completed.")

video_path = "/Users/niyaz/Downloads/test_densepose.mp4"  # Update this path to your video file
process_video(video_path)
