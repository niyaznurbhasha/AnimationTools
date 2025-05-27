import cv2
import numpy as np
import mediapipe as mp
import argparse
import os

# Convert the extracted landmarks from MediaPipe to 68-point format
def convert_ldmk_to_68(mediapipe_ldmk):
    return np.stack([
        mediapipe_ldmk[:, 234], mediapipe_ldmk[:, 93], mediapipe_ldmk[:, 132], mediapipe_ldmk[:, 58],
        mediapipe_ldmk[:, 172], mediapipe_ldmk[:, 136], mediapipe_ldmk[:, 150], mediapipe_ldmk[:, 176],
        mediapipe_ldmk[:, 152], mediapipe_ldmk[:, 400], mediapipe_ldmk[:, 379], mediapipe_ldmk[:, 365],
        mediapipe_ldmk[:, 397], mediapipe_ldmk[:, 288], mediapipe_ldmk[:, 361], mediapipe_ldmk[:, 323],
        mediapipe_ldmk[:, 454], mediapipe_ldmk[:, 70], mediapipe_ldmk[:, 63], mediapipe_ldmk[:, 105],
        mediapipe_ldmk[:, 66], mediapipe_ldmk[:, 107], mediapipe_ldmk[:, 336], mediapipe_ldmk[:, 296],
        mediapipe_ldmk[:, 334], mediapipe_ldmk[:, 293], mediapipe_ldmk[:, 300], mediapipe_ldmk[:, 168],
        mediapipe_ldmk[:, 6], mediapipe_ldmk[:, 195], mediapipe_ldmk[:, 4], mediapipe_ldmk[:, 239],
        mediapipe_ldmk[:, 241], mediapipe_ldmk[:, 19], mediapipe_ldmk[:, 461], mediapipe_ldmk[:, 459],
        mediapipe_ldmk[:, 33], mediapipe_ldmk[:, 160], mediapipe_ldmk[:, 158], mediapipe_ldmk[:, 133],
        mediapipe_ldmk[:, 153], mediapipe_ldmk[:, 144], mediapipe_ldmk[:, 362], mediapipe_ldmk[:, 385],
        mediapipe_ldmk[:, 387], mediapipe_ldmk[:, 263], mediapipe_ldmk[:, 373], mediapipe_ldmk[:, 380],
        mediapipe_ldmk[:, 61], mediapipe_ldmk[:, 40], mediapipe_ldmk[:, 37], mediapipe_ldmk[:, 0],
        mediapipe_ldmk[:, 267], mediapipe_ldmk[:, 270], mediapipe_ldmk[:, 291], mediapipe_ldmk[:, 321],
        mediapipe_ldmk[:, 314], mediapipe_ldmk[:, 17], mediapipe_ldmk[:, 84], mediapipe_ldmk[:, 91],
        mediapipe_ldmk[:, 78], mediapipe_ldmk[:, 81], mediapipe_ldmk[:, 13], mediapipe_ldmk[:, 311],
        mediapipe_ldmk[:, 308], mediapipe_ldmk[:, 402], mediapipe_ldmk[:, 14], mediapipe_ldmk[:, 178]
    ], axis=1)

# Helper function to calculate pose (rotation matrix and translation vector)
def calculate_pose(landmarks_3d):
    eye_left = landmarks_3d[33]
    eye_right = landmarks_3d[263]
    nose_tip = landmarks_3d[1]
    chin = landmarks_3d[152]

    forward_vector = chin - nose_tip
    forward_vector = forward_vector / np.linalg.norm(forward_vector)
    
    right_vector = eye_right - eye_left
    right_vector = right_vector / np.linalg.norm(right_vector)

    up_vector = np.cross(right_vector, forward_vector)

    # Build rotation matrix
    rotation_matrix = np.column_stack((right_vector, up_vector, forward_vector))
    translation_vector = nose_tip  # Use nose tip as reference for translation

    # Create transformation matrix (4x4)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector

    return transformation_matrix

# Main function to extract landmarks and pose from video using MediaPipe
def extract_landmarks_and_pose_from_video(video_path, output_dir):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

    video_cap = cv2.VideoCapture(video_path)
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    all_landmarks = []
    all_poses = []

    while video_cap.isOpened():
        success, frame = video_cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # Extract 3D landmarks
            landmarks_3d = np.array([(landmark.x, landmark.y, landmark.z) for landmark in face_landmarks.landmark])

            # Scale landmarks to match image resolution
            landmarks_3d[:, 0] *= width
            landmarks_3d[:, 1] *= height

            # Get the 68-point landmarks for consistency
            landmarks_68 = convert_ldmk_to_68(landmarks_3d[np.newaxis, ...])
            all_landmarks.append(landmarks_68[0])

            # Pose calculation
            pose_matrix = calculate_pose(landmarks_3d)
            all_poses.append(pose_matrix)

    video_cap.release()
    face_mesh.close()

    # Convert list of landmarks and poses to numpy arrays
    all_landmarks = np.stack(all_landmarks, axis=0)
    all_poses = np.stack(all_poses, axis=0)

    # Save the landmarks and poses
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'landmarks.npy'), all_landmarks)
    np.save(os.path.join(output_dir, 'poses.npy'), all_poses)

    print(f"Landmarks and poses saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 68-point landmarks and pose from a video using MediaPipe.")
    parser.add_argument("video_path", type=str, help="Path to the input video.")
    parser.add_argument("output_dir", type=str, help="Directory to save the landmarks and pose.")
    args = parser.parse_args()

    extract_landmarks_and_pose_from_video(args.video_path, args.output_dir)
