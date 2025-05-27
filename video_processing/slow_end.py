import cv2
import argparse

def slow_down_last_segment(input_file, output_file, slow_duration, slow_factor):
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print("Error: Cannot open input video.")
        return

    # Retrieve video properties.
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Determine how many frames correspond to the slow duration.
    slow_frames = int(slow_duration * fps)
    # Ensure we don't exceed available frames.
    start_slow_idx = max(total_frames - slow_frames, 0)

    # Setup VideoWriter. The output fps remains the same.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # For frames before the slow segment, write normally.
        if frame_idx < start_slow_idx:
            out.write(frame)
        else:
            # For the slow segment, duplicate each frame by slow_factor.
            for _ in range(slow_factor):
                out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Processing complete. Output saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Slow down the last X seconds of a video by duplicating frames."
    )
    parser.add_argument("--input", required=True, help="Path to the input video file")
    parser.add_argument("--output", required=True, help="Path to the output video file")
    parser.add_argument("--slow_duration", type=float, required=True,
                        help="Duration (in seconds) at the end of the video to slow down")
    parser.add_argument("--slow_factor", type=int, default=2,
                        help="Slowdown factor (e.g., 2 means double the duration of the last segment). Default is 2.")

    args = parser.parse_args()
    slow_down_last_segment(args.input, args.output, args.slow_duration, args.slow_factor)
