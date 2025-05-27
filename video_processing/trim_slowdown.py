import subprocess
import sys
def process_video(input_file, output_file, start_trim, end_trim, slowdown_factor):
    import subprocess

    # Get the total duration
    cmd_get_duration = [
        "ffprobe", "-i", input_file, "-show_entries", "format=duration",
        "-v", "quiet", "-of", "csv=p=0"
    ]
    duration = float(subprocess.check_output(cmd_get_duration).decode().strip())

    # Calculate end time after trimming
    end_time = duration - end_trim

    # Trim the video
    trimmed_file = "trimmed.mp4"
    cmd_trim = [
        "ffmpeg", "-i", input_file, "-ss", str(start_trim), "-to", str(end_time),
        "-c", "copy", trimmed_file
    ]
    subprocess.run(cmd_trim, check=True)

    # Check if audio exists
    cmd_check_audio = [
        "ffprobe", "-i", trimmed_file, "-show_streams", "-select_streams", "a",
        "-loglevel", "error"
    ]
    has_audio = subprocess.run(cmd_check_audio, capture_output=True, text=True).stdout.strip()

    # Construct the ffmpeg slowdown command
    if has_audio:
        cmd_slowdown = [
            "ffmpeg", "-i", trimmed_file, "-filter_complex",
            f"[0:v]setpts={slowdown_factor}*PTS[v];[0:a]atempo={1/slowdown_factor}[a]",
            "-map", "[v]", "-map", "[a]", output_file
        ]
    else:
        cmd_slowdown = [
            "ffmpeg", "-i", trimmed_file, "-filter_complex",
            f"[0:v]setpts={slowdown_factor}*PTS[v]",
            "-map", "[v]", "-c:v", "libx264", "-an", output_file  # Disable audio if missing
        ]

    subprocess.run(cmd_slowdown, check=True)
    print(f"Processed video saved as {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python script.py input.mp4 output.mp4 start_trim end_trim slowdown_factor")
        sys.exit(1)
    
    input_mp4 = sys.argv[1]
    output_mp4 = sys.argv[2]
    start_trim = float(sys.argv[3])
    end_trim = float(sys.argv[4])
    slowdown_factor = float(sys.argv[5])
    
    process_video(input_mp4, output_mp4, start_trim, end_trim, slowdown_factor)
