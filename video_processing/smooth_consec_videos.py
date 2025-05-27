import cv2
import numpy as np

def resize_and_pad(frame, target_size=(1080, 1920)):
    """
    Resize and pad the frame to the target size while preserving aspect ratio.
    """
    target_w, target_h = target_size
    h, w = frame.shape[:2]
    
    # Compute the scaling factor while preserving aspect ratio
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize with aspect ratio preserved
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create a black canvas of the target size
    padded_frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Center the resized image on the canvas
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    padded_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame
    
    return padded_frame

def alpha_blend(frame1, frame2, alpha):
    """
    Simple alpha blending: 
    output = frame1*(1-alpha) + frame2*(alpha)
    """
    return cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)

def smooth_transition(video1_path, video2_path, output_path, 
                      transition_frames=30,  # how many frames of transition
                      target_size=(1080, 1920)):

    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    # Fallback to 30 FPS if the video doesn't give a valid FPS
    fps1 = cap1.get(cv2.CAP_PROP_FPS) or 30
    fps2 = cap2.get(cv2.CAP_PROP_FPS) or 30

    # Choose an output FPS. You can pick max(fps1, fps2) if you prefer
    out_fps = int(fps1)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, out_fps, target_size)

    # Read & store frames from video 1
    frames1 = []
    while True:
        ret, frame = cap1.read()
        if not ret:
            break
        frame_resized = resize_and_pad(frame, target_size)
        frames1.append(frame_resized)

    # Read & store frames from video 2
    frames2 = []
    while True:
        ret, frame = cap2.read()
        if not ret:
            break
        frame_resized = resize_and_pad(frame, target_size)
        frames2.append(frame_resized)

    cap1.release()
    cap2.release()

    if not frames1 or not frames2:
        print("Error: One or both videos have no frames.")
        return

    # 1) Write all frames of video 1
    for f in frames1:
        out.write(f)

    # 2) Transition from the *last frame of video 1* to the *first frame of video 2*
    last_frame_v1 = frames1[-1]
    first_frame_v2 = frames2[0]
    
    for i in range(transition_frames):
        alpha = i / float(transition_frames - 1)  # alpha goes from 0.0 to 1.0
        blended = alpha_blend(last_frame_v1, first_frame_v2, alpha)
        out.write(blended)

    # 3) Write all frames of video 2
    for f in frames2:
        out.write(f)

    out.release()
    print(f"Output video saved to: {output_path}")


# Example usage:
smooth_transition("heroReveal_04.mp4",
                  "herostand_vertical_05.mp4",
                  "output.mp4",
                  transition_frames=5,
                  target_size=(1080, 1920))
