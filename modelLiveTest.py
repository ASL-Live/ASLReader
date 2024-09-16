import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('model.h5')

# Actions the model was trained on
actions = np.array(['busy', 'deaf', 'excuse me', 'fine', 'good', 'hard of hearing', 'hearing','help'])

# Set up MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

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

# Function to perform mediapipe detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert the color from BGR to RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make predictions
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert the color back to BGR
    return image, results

# Function to draw landmarks
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# Function to extract keypoints and reduce dimensionality
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])

# Fixed resolution for the camera feed
FIXED_RESOLUTION = (640, 480)  # Example resolution (width, height)

# Initialize variables for real-time detection
sequence = []
predictions = []
threshold = 0.7
SEQUENCE_LENGTH = 100  # The model expects 100 frames per sequence
detected_action = ""  # Variable to store the last detected action

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Resize the frame to the fixed resolution
        frame = cv2.resize(frame, FIXED_RESOLUTION)

        # Perform detection and draw landmarks
        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)

        # Check if hands are detected
        if results.left_hand_landmarks or results.right_hand_landmarks:
            # Extract keypoints and make predictions
            keypoints = extract_keypoints(results)

            # Augment and normalize keypoints
            augmented_keypoints = augment_keypoints(keypoints)
            normalized_keypoints = normalize_keypoints(augmented_keypoints)

            sequence.append(normalized_keypoints)

            # Ensure the sequence is exactly SEQUENCE_LENGTH (100 frames)
            if len(sequence) > SEQUENCE_LENGTH:
                sequence = sequence[-SEQUENCE_LENGTH:]  # Keep the last 100 frames

            # If fewer than SEQUENCE_LENGTH frames, pad the sequence by repeating the last frame
            if len(sequence) < SEQUENCE_LENGTH:
                while len(sequence) < SEQUENCE_LENGTH:
                    sequence.append(sequence[-1])

            # Only predict if the sequence length matches the model's expected input length
            if len(sequence) == SEQUENCE_LENGTH:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))

                confidence = res[np.argmax(res)]
                action = actions[np.argmax(res)]

                if confidence > threshold:
                    detected_action = action  # Store the detected action

        # Display the detected action on the video feed
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)  # Background for action display
        cv2.putText(image, detected_action, (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the video feed
        cv2.imshow('Feed', image)

        # Exit on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
