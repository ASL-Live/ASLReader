import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('model.h5')

# Actions the model was trained on
actions = np.array(['hello', 'thank you'])

# Set up MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


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
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([pose, lh, rh])


# Initialize variables for real-time detection
sequence = []
sentence = []
predictions = []
threshold = 0.8

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Set up the MediaPipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Perform detection and draw landmarks
        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)

        # Extract keypoints and make predictions
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))

            confidence = res[np.argmax(res)]
            action = actions[np.argmax(res)]

            if confidence > threshold:
                if len(sentence) == 0 or (sentence[-1] != action):
                    sentence.append(action)

                # Print the detected word and its confidence to the terminal
                print(f"Detected: {action} with confidence {confidence:.2f}")

        # Display the detected action
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the video feed
        cv2.imshow('Feed', image)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
