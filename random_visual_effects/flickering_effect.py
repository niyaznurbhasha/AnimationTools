import cv2
import numpy as np
import random

def load_images(paths):
    images = [cv2.imread(path) for path in paths]
    return images

def create_flicker_video_with_flashes(images, output_path, duration=10, fps=24, flash_duration=2, flash_chance=0.1):
    """
    Creates a video with a primary image displayed most of the time and brief flashes of other images.
    
    :param images: List of images where the first image is the primary one.
    :param output_path: Path to save the output video.
    :param duration: Total duration of the video in seconds.
    :param fps: Frames per second of the output video.
    :param flash_duration: Duration of each flash in frames (not seconds).
    :param flash_chance: Chance of a flash occurring on a given frame.
    """
    height, width, layers = images[0].shape
    size = (width, height)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    num_frames = duration * fps
    primary_image = images[0]
    
    for frame in range(num_frames):
        # Randomly decide if this frame will start a flash
        if random.random() < flash_chance and len(images) > 1:
            flash_image = random.choice(images[1:])  # Choose one of the non-primary images
            for _ in range(flash_duration):
                if frame < num_frames:  # Check to avoid going over the total frame count
                    out.write(flash_image)
                    frame += 1
        else:
            out.write(primary_image)
    
    out.release()

# Paths to your images
image_paths = ['/Users/niyaz/Downloads/char1_3pose_2.png', '/Users/niyaz/Documents/foo.png', '/Users/niyaz/Documents/foo3.png']
images = load_images(image_paths)

# Output path for the video
output_video_path = 'flicker_effect.mp4'

# Parameters
duration = 10  # total video duration in seconds
fps = 24  # frames per second
flash_duration = int(0.2 * fps)  # flash duration in frames (e.g., 0.2 seconds)
flash_chance = 0.05  # chance of a flash occurring on a given frame

# Create the flickering effect video
create_flicker_video_with_flashes(images, output_video_path, duration, fps, flash_duration, flash_chance)

print("Video saved to:", output_video_path)

