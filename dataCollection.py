import cv2
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
import time
import mediapipe as mp

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    # mp_drawing.draw_landmarks(image, results.face_landmarks)  # This line is removed
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

# The rest of the script remains the same


DATA_PATH = os.path.join('MP_Data')
VIDEOS_PATH = 'videos'  # Path to the folder containing your videos

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Using glob to find all .mp4 files in the videos folder
video_files = glob.glob(os.path.join(VIDEOS_PATH, '*.mp4'))

# Process each video file
for video_file in video_files:
    # Open video file
    cap = cv2.VideoCapture(video_file)
    action = os.path.basename(video_file).split('.')[0]  # Extract action name from file name

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        sequence = 0  # If each video is considered a separate sequence
        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Break the loop if no frames are returned

            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)

            keypoints = extract_keypoints(results)
            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
            os.makedirs(os.path.dirname(npy_path), exist_ok=True)
            np.save(npy_path, keypoints)

            cv2.imshow('Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            frame_num += 1

        cap.release()

cv2.destroyAllWindows()
