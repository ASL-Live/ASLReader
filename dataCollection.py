import cv2
import numpy as np
import os
import glob
import mediapipe as mp


# Augment keypoints by adding random noise
def augment_keypoints(keypoints):
    noise = np.random.normal(0, 0.01, keypoints.shape)
    return keypoints + noise


# Normalize the keypoints
def normalize_keypoints(keypoints):
    return (keypoints - np.mean(keypoints)) / np.std(keypoints)


# Smooth the keypoints using a moving average
def smooth_keypoints(keypoints_list, window_size=5):
    smoothed_keypoints = []
    for i in range(len(keypoints_list)):
        start = max(0, i - window_size // 2)
        end = min(len(keypoints_list), i + window_size // 2 + 1)
        smoothed_keypoints.append(np.mean(keypoints_list[start:end], axis=0))
    return smoothed_keypoints


# Perform MediaPipe detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


# Draw the landmarks on the image
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


# Extract keypoints for pose, left hand, and right hand
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([pose, lh, rh])


# Paths for the data and videos
DATA_PATH = os.path.join('MP_Data')
VIDEOS_PATH = 'videos'  # Path to the folder containing your videos

# Fixed resolution for all videos
FIXED_RESOLUTION = (640, 480)  # Example resolution (width, height)

# Hyperparameters for tuning
SEQUENCE_LENGTH = 200  # You can experiment with this for longer sequences

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
            with mp_holistic.Holistic(min_detection_confidence=0.7,
                                      min_tracking_confidence=0.7) as holistic:  # Increased confidence thresholds
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break  # Break the loop if no frames are returned

                    # Resize the frame to the fixed resolution
                    frame = cv2.resize(frame, FIXED_RESOLUTION)

                    # Perform detection and extract results
                    image, results = mediapipe_detection(frame, holistic)
                    draw_landmarks(image, results)

                    keypoints = extract_keypoints(results)

                    # Augment keypoints
                    augmented_keypoints = augment_keypoints(keypoints)

                    # Normalize keypoints
                    normalized_keypoints = normalize_keypoints(augmented_keypoints)

                    # Append normalized keypoints to the list
                    keypoints_list.append(normalized_keypoints)

                    cv2.imshow('Feed', image)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

            # Smooth keypoints to reduce noise
            #keypoints_list = smooth_keypoints(keypoints_list)

            # Downsample the sequence if it has more than the specified sequence length
            if len(keypoints_list) > SEQUENCE_LENGTH:
                indices = np.linspace(0, len(keypoints_list) - 1, SEQUENCE_LENGTH).astype(int)
                keypoints_list = [keypoints_list[i] for i in indices]

            # If the sequence has fewer than the specified sequence length, duplicate the last frame
            while len(keypoints_list) < SEQUENCE_LENGTH:
                keypoints_list.append(keypoints_list[-1])  # Duplicate the last frame

            # Save the keypoints
            for frame_num, keypoints in enumerate(keypoints_list):
                npy_path = os.path.join(DATA_PATH, action_folder, str(sequence), str(frame_num))
                os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                np.save(npy_path, keypoints)

            cap.release()

cv2.destroyAllWindows()
