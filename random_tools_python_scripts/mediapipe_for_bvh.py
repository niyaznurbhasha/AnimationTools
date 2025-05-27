import cv2
import mediapipe as mp
import numpy as np
import json
import os

mp_holistic = mp.solutions.holistic

def extract_holistic_keypoints(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    frame_index = 0
    keypoints_all_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        
        keypoints = {}
        
        if results.pose_landmarks:
            keypoints['pose'] = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
        
        if results.left_hand_landmarks:
            keypoints['left_hand'] = [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
        
        if results.right_hand_landmarks:
            keypoints['right_hand'] = [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
        
        keypoints_all_frames.append(keypoints)
        
        frame_index += 1
    
    cap.release()
    
    for i, keypoints in enumerate(keypoints_all_frames):
        with open(os.path.join(output_dir, f'frame_{i:06d}.json'), 'w') as f:
            json.dump(keypoints, f)

video_path = 'path_to_your_video.mp4'
output_dir = 'keypoints_output'
extract_holistic_keypoints(video_path, output_dir)
