import os
import subprocess

def combine_videos_with_audio(folder_path, output_video="final_output.mp4"):
    videos = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.mp4', '.mov', '.avi', '.mkv'))])
    audio_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.mp3', '.wav', '.aac'))]

    if not videos:
        print("No video files found.")
        return
    if not audio_files:
        print("No audio file found.")
        return

    audio_file = audio_files[0]  # Assume only one audio file

    # Step 1: Concatenate videos
    temp_video = os.path.join(folder_path, "temp_output.mp4")
    
    command = ["ffmpeg"]
    for video in videos:
        command.extend(["-i", video])

    filter_complex = "".join([f"[{i}:v:0]" for i in range(len(videos))]) + f"concat=n={len(videos)}:v=1[outv]"

    concat_command = command + [
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23", temp_video
    ]

    subprocess.run(concat_command, check=True)

    # Step 2: Replace audio
    final_command = [
        "ffmpeg", "-i", temp_video, "-i", audio_file, "-c:v", "copy",
        "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-strict", "experimental", output_video
    ]

    subprocess.run(final_command, check=True)

    # Cleanup
    os.remove(temp_video)

    print(f"Final video saved as {output_video}")

# Usage
combine_videos_with_audio("post2_vids")
