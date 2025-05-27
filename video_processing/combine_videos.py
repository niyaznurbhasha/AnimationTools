import os
import cv2
import numpy as np
import subprocess

def resize_and_pad(frame, target_size=(1080, 1920)):
    target_w, target_h = target_size
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded_frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    padded_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame
    return padded_frame

def adjust_brightness(frame, factor=1.0):
    return cv2.convertScaleAbs(frame, alpha=factor, beta=0)

def crossfade_transition_frames(frames1, frames2):
    T = min(len(frames1), len(frames2))
    transition_list = []
    for i in range(T):
        alpha = i / (T - 1) if T > 1 else 0
        blended = cv2.addWeighted(frames1[i], 1 - alpha, frames2[i], alpha, 0)
        transition_list.append(blended)
    return transition_list

def blur_crossfade_transition_frames(frames1, frames2, max_kernel=21):
    T = min(len(frames1), len(frames2))
    transition_list = []
    for i in range(T):
        alpha = i / (T - 1) if T > 1 else 0
        blur_factor = 1 - abs(alpha - 0.5) * 2  # 0 at edges, 1 at center
        kernel_size = max(1, int(blur_factor * max_kernel))
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size > 1:
            frame1_blur = cv2.GaussianBlur(frames1[i], (kernel_size, kernel_size), 0)
            frame2_blur = cv2.GaussianBlur(frames2[i], (kernel_size, kernel_size), 0)
        else:
            frame1_blur = frames1[i]
            frame2_blur = frames2[i]
        blended = cv2.addWeighted(frame1_blur, 1 - alpha, frame2_blur, alpha, 0)
        transition_list.append(blended)
    return transition_list

def combine_processed_videos(video_paths, output_path, fps, target_size, transition_frames=15, transition_type="blur"):
    """
    Combines the processed videos into one output file with overlapping transitions.
    
    For each pair of consecutive videos:
      - The last `transition_frames` of the previous clip and the first `transition_frames`
        of the next clip are blended together.
    For the last video, a transition is generated (if possible) and then the entire clip is appended,
    ensuring no frames are lost.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)
    
    videos_frames = []
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        videos_frames.append(frames)
    
    final_frames = []
    
    # Process the first video: append all but the last transition_frames (if available).
    first_vid = videos_frames[0]
    if len(first_vid) > transition_frames:
        final_frames.extend(first_vid[:-transition_frames])
        prev_overlap = first_vid[-transition_frames:]
    else:
        final_frames.extend(first_vid)
        prev_overlap = first_vid
    
    # Process intermediate videos (if any)
    for i in range(1, len(videos_frames) - 1):
        curr_vid = videos_frames[i]
        if len(curr_vid) >= transition_frames:
            curr_overlap = curr_vid[:transition_frames]
            if transition_type.lower() == "blur":
                transition_seq = blur_crossfade_transition_frames(prev_overlap, curr_overlap)
            else:
                transition_seq = crossfade_transition_frames(prev_overlap, curr_overlap)
            final_frames.extend(transition_seq)
            # Append the main part of the video (skipping the overlapping parts)
            if len(curr_vid) > 2 * transition_frames:
                final_frames.extend(curr_vid[transition_frames:-transition_frames])
                prev_overlap = curr_vid[-transition_frames:]
            else:
                final_frames.extend(curr_vid[transition_frames:])
                prev_overlap = curr_vid[-transition_frames:]
        else:
            final_frames.extend(curr_vid)
            prev_overlap = curr_vid
    
    # Process the last video: generate a transition then append the entire clip.
    last_vid = videos_frames[-1]
    if len(last_vid) >= transition_frames:
        curr_overlap = last_vid[:transition_frames]
        if transition_type.lower() == "blur":
            transition_seq = blur_crossfade_transition_frames(prev_overlap, curr_overlap)
        else:
            transition_seq = crossfade_transition_frames(prev_overlap, curr_overlap)
        final_frames.extend(transition_seq)
    # Append the entire last video (preserving all its frames)
    final_frames.extend(last_vid)
    
    for frame in final_frames:
        out.write(frame)
    out.release()
    print(f"Combined video with transitions saved as '{output_path}'.")

def add_audio_to_video(video_path, audio_path, output_path):
    command = [
        "ffmpeg", "-y", 
        "-i", video_path, 
        "-i", audio_path, 
        "-c:v", "copy", 
        "-c:a", "aac", "-strict", "experimental",
        "-shortest", 
        output_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"Final video with audio saved as '{output_path}'.")

def process_videos_in_folder(folder_path, 
                             brightness_factors=None, 
                             global_brightness=None, 
                             output_folder="output_videos",
                             target_size=(1080, 1920),
                             combine_videos=False,
                             combined_output="final_combined_video.mp4",
                             transition_frames=15,
                             transition_type="blur",
                             background_audio=None):
    os.makedirs(output_folder, exist_ok=True)
    
    video_files = sorted([
        f for f in os.listdir(folder_path) 
        if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))
    ])
    
    if not video_files:
        print("No video files found in the folder.")
        return
    
    if brightness_factors is None:
        brightness_factors = [global_brightness if global_brightness is not None else 1.0] * len(video_files)
    if len(brightness_factors) < len(video_files):
        brightness_factors += [global_brightness if global_brightness is not None else 1.0] * (len(video_files) - len(brightness_factors))
    
    processed_video_paths = []
    common_fps = None  
    
    for idx, video_file in enumerate(video_files):
        input_path = os.path.join(folder_path, video_file)
        brightness_factor = brightness_factors[idx]
        
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        if common_fps is None:
            common_fps = fps
        w_out, h_out = target_size
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}_processed.mp4")
        out_writer = cv2.VideoWriter(out_path, fourcc, fps, (w_out, h_out))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_processed = resize_and_pad(frame, target_size)
            frame_processed = adjust_brightness(frame_processed, brightness_factor)
            out_writer.write(frame_processed)
        
        cap.release()
        out_writer.release()
        
        processed_video_paths.append(out_path)
        print(f"Processed '{video_file}' → '{out_path}' with brightness factor {brightness_factor}.\n")
    
    if combine_videos and processed_video_paths:
        combined_video_path = os.path.join(output_folder, combined_output)
        combine_processed_videos(
            processed_video_paths, 
            combined_video_path,
            common_fps, 
            target_size, 
            transition_frames,
            transition_type
        )
        if background_audio:
            final_video_with_audio = os.path.join(output_folder, "final_with_audio.mp4")
            add_audio_to_video(combined_video_path, background_audio, final_video_with_audio)

    print("All videos processed!")

# -------------- EXAMPLE USAGE --------------
if __name__ == "__main__":
    folder_to_process = "post2_vids"  # Replace with your folder path
    global_brightness = 0.8       # (1.0 = no change)
 #   brightness_list = [1.2, 0.8]       # Per-video brightness factors (alphabetical order)
 #   background_audio = "background_music.mp3"  # Optional audio file
    
    process_videos_in_folder(
        folder_path=folder_to_process,
        brightness_factors=None,
        global_brightness=global_brightness,
        output_folder="output_videos",
        target_size=(1080, 1920),
        combine_videos=True,
        combined_output="final_combined_video.mp4",
        transition_frames=15,
        transition_type="blur",
        background_audio=None,
    )
