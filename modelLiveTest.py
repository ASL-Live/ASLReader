import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('model.h5')

# Load global mean and std
global_mean = np.load('global_mean.npy')
global_std = np.load('global_std.npy')

def normalize_keypoints(keypoints):
    return (keypoints - global_mean) / global_std

# Actions the model was trained on
actions = np.array(['busy', 'deaf', 'excuse me', 'fine',])

# Set up MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Function to perform mediapipe detection
def mediapipe_detection(image, holistic_model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert the color from BGR to RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = holistic_model.process(image)  # Make predictions
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert the color back to BGR
    return image, results

# Function to draw landmarks
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# Function to extract keypoints and reduce dimensionality
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])

# Fixed resolution for webcam feed
FIXED_RESOLUTION = (640, 480)  # Width, Height

# Initialize variables for real-time detection
sequence = []
predictions = []
threshold = 0.7
SEQUENCE_LENGTH = 100  # The model expects 100 frames per sequence
detected_action = ""  # Variable to store the last detected action

# Open the webcam (usually device index 0)
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

        # Extract keypoints and make predictions
        keypoints = extract_keypoints(results)

        # Normalize keypoints using global mean and std
        normalized_keypoints = normalize_keypoints(keypoints)

        sequence.append(normalized_keypoints)

        # Ensure the sequence is exactly SEQUENCE_LENGTH frames
        if len(sequence) > SEQUENCE_LENGTH:
            sequence = sequence[-SEQUENCE_LENGTH:]  # Keep the last SEQUENCE_LENGTH frames

        # Only predict if the sequence length matches the model's expected input length
        if len(sequence) == SEQUENCE_LENGTH:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            confidence = res[np.argmax(res)]
            action = actions[np.argmax(res)]

            # Append prediction and check if confidence is above threshold
            predictions.append(np.argmax(res))
            if confidence > threshold:
                detected_action = action  # Update detected action

        # Display the detected action on the video feed
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)  # Background for action display
        cv2.putText(image, detected_action, (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # Show the video feed
        cv2.imshow('Webcam Feed', image)

        # Exit on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
