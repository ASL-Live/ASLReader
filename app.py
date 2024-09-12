from flask import Flask,request,jsonify, render_template
from keras.models import load_model
import pickle
import mediapipe as mp
import numpy as np
import cv2
############Short Documentation
#-How It Works
#   Receives video from front-end as soon as user clicks 'Demo' Page
#   Uses video Data received and stores it in a variable that is accessible by the model
#   Feeds Data to the model, makes prediction, and returns the data to front-end in JSON format


app = Flask(__name__)

#load trained model
model = pickle.load(open('model.pkl', 'rb'))
sequence = []
predictions = []
threshold = 0.8
actions = np.array(['hello', 'please', 'thank you'])

mp_holistic = mp.solutions.holistic
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert the color from BGR to RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make predictions
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert the color back to BGR
    return image, results
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([pose, lh, rh])


### Defining app routes###
#Home page route to connect ML backend to our webpage
@app.route('/')
def home():
    return render_template('ASLive\src\App.jsx')

#ASL-live.com/demo
#Using existing method to open the camera and generate frames from the video
#Then returns the predictions and the confidence scores
@app.route('/demo', methods = ['POST'])
def video_feed():
    #capture live video from webcam
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.release()
                cv2.destroyAllWindows()
                return jsonify({'error': 'Failed to grab frame'}), 500

            # Perform detection and draw landmarks
            image, results = mediapipe_detection(frame, holistic)

            # Check if hands are detected
            if results.left_hand_landmarks or results.right_hand_landmarks:
                # Extract keypoints and make predictions
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]

    if len(sequence) == 30:
        #feed results of the live webcam recording to the model
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        #store results in predictions variable
        predictions.append(np.argmax(res))

        if not predictions:
            return jsonify({'error': 'Failed to generate predictions'})

        confidence = res[np.argmax(res)]
        action = actions[np.argmax(res)]

        if confidence > threshold:
            # Response format
            response_data = {
                'predictions': predictions,
                'confidence': float(confidence)  # Convert to float for JSON serialization
            }

            # Return the data as JSON
            return jsonify(response_data)

    cap.release()
    cv2.destroyAllWindows()
    return jsonify({'error': 'No prediction available'}), 404


if __name__ == "__main__":
     app.run(debug=True)
    