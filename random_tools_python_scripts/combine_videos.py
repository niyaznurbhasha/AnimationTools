from moviepy.editor import VideoFileClip, concatenate_videoclips,vfx

import os

# Path to the root directory containing all the subfolders
root_dir = "/Users/niyaz/Documents/animate-anything/output/demo/"
# Path to the output directory where the combined videos will be saved
output_dir = os.path.join(root_dir, "combined_videos")

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Initialize a counter for naming the output files
output_file_counter = 1

# Loop through each subfolder in the root directory
for subfolder in os.listdir(root_dir):
    subfolder_path = os.path.join(root_dir, subfolder)
    if os.path.isdir(subfolder_path):
        # List to store the clips for this subfolder
        clips = []

        # Load each of the 6 videos (0.mp4 to 5.mp4) and append them to the clips list
        for i in range(6):
            video_path = os.path.join(subfolder_path, f"{i}.mp4")
            if os.path.exists(video_path):
                clip = VideoFileClip(video_path)
                clips.append(clip)

        # Concatenate the clips
        final_clip = concatenate_videoclips(clips)

               # Double the frame rate of the combined clip
        # Note: moviepy does not directly change the frame rate in a way that also halves the duration,
        # so we speed up the clip, which doubles the frame rate and halves the duration.
        final_clip = final_clip.fx(vfx.speedx, 2)

        # Output file path for the combined video
        output_file_path = os.path.join(output_dir, f"{output_file_counter}.mp4")
        output_file_counter += 1

        # Write the result to the file
        final_clip.write_videofile(output_file_path)
        print(f"Combined video saved to {output_file_path}")

print("All videos have been processed.")
exit()
from moviepy.editor import VideoFileClip, concatenate_videoclips
import re
import os

# Directory where your clips are stored
directory = "/Users/niyaz/Documents/magic-animate/samples/animation-2024-02-21T09-10-08/videos/"

# Regex to match files like "main2_robot_output_part1.mp4"
pattern = r'main2_robot_output_part(\d+).mp4'

# Collect and sort the video file paths
video_files = []
for filename in os.listdir(directory):
    match = re.match(pattern, filename)
    if match:
        # Store the file path and the part number for sorting
        video_files.append((filename, int(match.group(1))))

# Sort the files by part number
video_files.sort(key=lambda x: x[1])

# Load the video files and concatenate them
clips = [VideoFileClip(os.path.join(directory, file)) for file, _ in video_files]
final_clip = concatenate_videoclips(clips)

# Output file path
output_path = os.path.join(directory, "combined_output2.mp4")

# Write the result to a file
final_clip.write_videofile(output_path)

print(f"Combined video saved to {output_path}")
