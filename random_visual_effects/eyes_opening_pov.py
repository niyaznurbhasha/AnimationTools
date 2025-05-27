import cv2
import numpy as np

def create_eye_mask(shape, center, size, opening):
    """
    Create an eye-shaped mask where `opening` controls the vertical size of the eye opening.
    """
    mask = np.zeros(shape[:2], dtype=np.uint8)
    axis = (int(size * 2), int(size * opening))
    cv2.ellipse(mask, center, axis, 0, 0, 360, (255, 255, 255), -1)
    return mask

def apply_eye_opening_effect(image_path, video_output_path='eye_opening_effect.mp4', fps=30, cycles=3):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"The specified image at {image_path} was not found.")
    height, width, channels = image.shape

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    # Opening sizes for each cycle and transition parameters
    opening_sizes = [0.3, 0.6, 1.0]  # Control how much the eye opens each cycle
    transition_start = fps * 2 * cycles  # Start of the transition
    transition_frames = int(fps * 1.5)  # Duration of the transition to full view

    total_frames = transition_start + transition_frames  # Total frames including transition

    for frame in range(total_frames):
        cycle_number = min(frame // (fps * 2), cycles - 1)
        cycle_pos = frame % (fps * 2)

        # Exponential opening effect
        if frame < transition_start:
            progress = cycle_pos / (fps * 2)  # Normalized progress [0, 1]
            growth_factor = 5  # Adjust for desired speed of opening
            opening_progress = (np.exp(growth_factor * progress) - 1) / (np.exp(growth_factor) - 1)
            opening = opening_progress * opening_sizes[cycle_number]
        else:
            opening = 1.0  # Keep fully open during the transition

        # Initial blur strength adjustment
        blur_strength = max(int((1 - opening) * 15), 1)
        blur_strength = blur_strength + 1 if blur_strength % 2 == 0 else blur_strength
        blurred_image = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)

        # Exponential decrease in blur during transition
        if frame >= transition_start:
            transition_progress = (frame - transition_start) / transition_frames
            k = 3  # Decay rate for the blur decrease
            A = 100  # Starting blur strength for the transition
            fade_strength = int(A * np.exp(-k * transition_progress))
            fade_strength = max(fade_strength, 1)
            fade_strength = fade_strength + 1 if fade_strength % 2 == 0 else fade_strength
            blurred_image = cv2.GaussianBlur(image, (fade_strength, fade_strength), 0)

        # Create and apply masks for both eyes
        mask_left = create_eye_mask((height, width), (width // 4, height // 2), 100, opening)
        mask_right = create_eye_mask((height, width), (3 * width // 4, height // 2), 100, opening)
        mask = cv2.bitwise_or(mask_left, mask_right)
        foreground = cv2.bitwise_and(blurred_image, blurred_image, mask=mask)
        if frame < transition_start:
            background = np.zeros((height, width, channels), dtype=np.uint8)
        else:
            background = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
        combined = cv2.add(foreground, background)

        out.write(combined)

    out.release()

# Example usage


# Example usage

image_path = "/Users/niyaz/Downloads/librarian1.png"  # Update this path
apply_eye_opening_effect(image_path)
