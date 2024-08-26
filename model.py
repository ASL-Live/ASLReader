import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

# Path for the exported data (numpy arrays)
DATA_PATH = os.path.join('MP_Data')

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Actions we want to detect
actions = np.array(['hello', 'thank you'])

label_map = {label: num for num, label in enumerate(actions)}

# Arrays to hold sequences and labels
sequences, labels = [], []

# Iterate over each action
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    sequences_folders = os.listdir(action_path)

    # Iterate over each sequence folder in the action directory
    for sequence_folder in sequences_folders:
        window = []
        sequence_path = os.path.join(action_path, sequence_folder)

        # Iterate over all frame files in the sequence folder
        for frame_file in sorted(os.listdir(sequence_path), key=lambda x: int(x.split('.')[0])):
            frame_path = os.path.join(sequence_path, frame_file)
            res = np.load(frame_path)
            window.append(res)

        sequences.append(window)
        labels.append(label_map[action])

# Set up data for training
x = np.array(sequences)
y = to_categorical(labels).astype(int)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Model creation
model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=True, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2]))))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(actions.shape[0], activation='softmax'))

# Set learning rate to train the model slower
learning_rate = 0.0001
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=170, callbacks=[tb_callback])

# Evaluate the model
yhat = model.predict(x_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

# Display evaluation metrics
print(multilabel_confusion_matrix(ytrue, yhat))
print(f'Accuracy: {accuracy_score(ytrue, yhat)}')

# Save the trained model
model.save('model.h5')
