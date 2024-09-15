import cv2
import numpy as np
import os
import glob
import mediapipe as mp

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

# Paths for the data and videos
DATA_PATH = os.path.join('MP_Data')
VIDEOS_PATH = 'videos'  # Path to the folder containing your videos

# Fixed resolution for all videos
FIXED_RESOLUTION = (640, 480)  # Example resolution (width, height)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Process each folder in the videos directory (each folder corresponds to a different action/word)
for action_folder in os.listdir(VIDEOS_PATH):
    action_path = os.path.join(VIDEOS_PATH, action_folder)
    if os.path.isdir(action_path):  # Ensure that it's a directory
        # Find all .mp4 files in the current action folder
        video_files = glob.glob(os.path.join(action_path, '*.mp4'))

        for video_file in video_files:
            cap = cv2.VideoCapture(video_file)
            sequence = os.path.basename(video_file).split('.')[0]  # Use the video filename as the sequence number

            keypoints_list = []
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break  # Break the loop if no frames are returned

                    # Resize the frame to the fixed resolution
                    frame = cv2.resize(frame, FIXED_RESOLUTION)

                    image, results = mediapipe_detection(frame, holistic)
                    draw_landmarks(image, results)

                    keypoints = extract_keypoints(results)
                    keypoints_list.append(keypoints)

                    cv2.imshow('Feed', image)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

            # Downsample the sequence if it has more than 60 frames
            if len(keypoints_list) > 60:
                indices = np.linspace(0, len(keypoints_list) - 1, 60).astype(int)
                keypoints_list = [keypoints_list[i] for i in indices]

            # If the sequence has fewer than 60 frames, duplicate the last frame
            while len(keypoints_list) < 60:
                keypoints_list.append(keypoints_list[-1])  # Duplicate the last frame

            # Save the keypoints
            for frame_num, keypoints in enumerate(keypoints_list):
                npy_path = os.path.join(DATA_PATH, action_folder, str(sequence), str(frame_num))
                os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                np.save(npy_path, keypoints)

            cap.release()

cv2.destroyAllWindows()
