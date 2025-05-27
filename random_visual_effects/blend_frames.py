import cv2
import numpy as np
import os
import re

def blend_frames(frame1, frame2, alpha=0.1):
    """
    Blends two frames together using a weighted sum.

    :param frame1: The first frame (as a NumPy array).
    :param frame2: The second frame (as a NumPy array).
    :param alpha: The weight for blending (0 to 1). Default is 0.1.
    :return: Blended frame.
    """
    return cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)

def process_video_frames(frame_folder, output_folder, alpha=0.1):
    """
    Process and blend frames from a folder and save the output.

    :param frame_folder: Folder containing the input frames.
    :param output_folder: Folder to save the blended frames.
    :param alpha: The weight for blending (0 to 1). Default is 0.1.
    """
    # Regular expression to match files like 'output_tintin_output_0001.png'
    file_pattern = re.compile(r'output_tintin_output_\d{4}\.png')

    # Sort files based on the numeric part of the filename
    frame_files = sorted([f for f in os.listdir(frame_folder) if file_pattern.match(f)],
                         key=lambda x: int(re.search(r'\d{4}', x).group()))

    previous_frame = None

    for i, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frame_folder, frame_file)
        output_path = os.path.join(output_folder, f"blended_frame_{i+1:04d}.png")

        # Read the current frame
        current_frame = cv2.imread(frame_path)

        # If it's the first frame, we don't blend it
        if previous_frame is None:
            blended_frame = current_frame
        else:
            # Blend with the previous frame
            blended_frame = blend_frames(previous_frame, current_frame, alpha)

        # Save the blended frame
        cv2.imwrite(output_path, blended_frame)

        # Update the previous frame
        previous_frame = current_frame

def moving_average_filter(frame_folder, output_folder, window_size=3):
    """
    Applies a moving average filter to the video frames.

    :param frame_folder: Directory containing the frames.
    :param output_folder: Directory where the filtered frames will be saved.
    :param window_size: Number of frames to include in the moving average (must be odd).
    """
    # Ensure the window size is odd
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")

    # Regular expression to match files like 'output_tintin_output_0001.png'
    file_pattern = re.compile(r'output_tintin_output_\d{4}\.png')

    # Sort files based on the numeric part of the filename
    frame_files = sorted([f for f in os.listdir(frame_folder) if file_pattern.match(f)],
                         key=lambda x: int(re.search(r'\d{4}', x).group()))

    # Pre-load frames for efficiency
    frames = [cv2.imread(os.path.join(frame_folder, f)) for f in frame_files]

    half_window = window_size // 2
    num_frames = len(frames)

    for i in range(num_frames):
        # Determine the start and end of the window
        start = max(i - half_window, 0)
        end = min(i + half_window + 1, num_frames)

        # Calculate the average of frames in the window
        avg_frame = np.mean(frames[start:end], axis=0).astype(np.uint8)

        # Save the averaged frame
        frame_number = re.search(r'\d{4}', frame_files[i]).group()
        output_filename = f"temporal_filtered_output_tintin_output_{frame_number}.png"
        cv2.imwrite(os.path.join(output_folder, output_filename), avg_frame)

def match_histograms(source_image, reference_image):
    """
    Adjust the color distribution of the source image to match that of the reference image.

    :param source_image: The image to be transformed.
    :param reference_image: The reference image with the desired color distribution.
    :return: The color-corrected image.
    """
    # Split into channels
    src_img_channels = cv2.split(source_image)
    ref_img_channels = cv2.split(reference_image)

    matched_channels = []

    for (src_channel, ref_channel) in zip(src_img_channels, ref_img_channels):
        # Calculate the histograms
        src_hist = cv2.calcHist([src_channel], [0], None, [256], [0, 256])
        ref_hist = cv2.calcHist([ref_channel], [0], None, [256], [0, 256])

        # Normalize the histograms
        src_hist_norm = src_hist.cumsum()
        src_hist_norm /= src_hist_norm[-1]
        ref_hist_norm = ref_hist.cumsum()
        ref_hist_norm /= ref_hist_norm[-1]

        # Create a lookup table
        lookup_table = np.searchsorted(ref_hist_norm, src_hist_norm)

        # Map the source channel pixels to the reference
        matched_channel = np.interp(src_channel.flatten(), np.arange(256), lookup_table).reshape(src_channel.shape)
        matched_channels.append(matched_channel.astype(np.uint8))

    # Merge the channels back together
    return cv2.merge(matched_channels)

def color_correct_frames(frame_folder, output_folder, reference_frame):
    """
    Perform color correction on frames in a folder based on a reference frame.

    :param frame_folder: Directory containing the frames.
    :param output_folder: Directory where the color-corrected frames will be saved.
    :param reference_frame: Filename of the reference frame.
    """
    # Regular expression to match files like 'output_tintin_output_0001.png'
    file_pattern = re.compile(r'output_tintin_output_\d{4}\.png')

    # Read the reference image
    reference_image = cv2.imread(os.path.join(frame_folder, reference_frame))

    # Process each frame
    for frame_file in sorted(os.listdir(frame_folder)):
        if file_pattern.match(frame_file):
            frame_path = os.path.join(frame_folder, frame_file)
            output_path = os.path.join(output_folder, frame_file)

            # Read the current frame
            current_frame = cv2.imread(frame_path)

            # Perform color correction
            color_corrected_frame = match_histograms(current_frame, reference_image)

            # Save the color-corrected frame
            cv2.imwrite(output_path, color_corrected_frame)

def blur_image(image_path, output_path, blur_size=(5, 5)):
    """
    Apply Gaussian blur to an image.

    :param image_path: Path to the input image.
    :param output_path: Path to save the blurred image.
    :param blur_size: Size of the Gaussian kernel. Default is (5, 5).
    """
    # Read the image
    image = cv2.imread(image_path)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (15,15), 0)

    # Save the blurred image
    cv2.imwrite(output_path, blurred_image)


# Example usage
image_path = '/Users/niyaz/Documents/neural-style-pt/examples/inputs/test1_style_lava/tintin.jpg'
output_path = '/Users/niyaz/Documents/neural-style-pt/examples/inputs/test1_style_lava/tintin_blurred.jpg'
blur_image(image_path, output_path)
exit()
# Example usage
frame_folder = 'path/to/your/frame_folder'
output_folder = 'path/to/your/output_folder'

# Example usage
frame_folder = '/Users/niyaz/Documents/neural-style-pt/examples/outputs/tintin_frames/'
output_folder = '/Users/niyaz/Documents/neural-style-pt/examples/outputs/tintin_frames_blended/'
reference_frame = 'output_tintin_output_0190.png'
#process_video_frames(frame_folder, output_folder, alpha=0.9)
#moving_average_filter(frame_folder, output_folder, window_size=25)
color_correct_frames(frame_folder, output_folder, reference_frame)
