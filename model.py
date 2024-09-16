import cv2
import numpy as np
import os
import mediapipe as mp
import pickle
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l2, l1_l2
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

def normalize_landmarks(landmarks, image_shape):
    return np.array([[res.x / image_shape[1], res.y / image_shape[0], res.z] for res in landmarks]).flatten()

def extract_keypoints(results):
    pose = normalize_landmarks(results.pose_landmarks.landmark, (480, 640)) if results.pose_landmarks else np.zeros(132)
    lh = normalize_landmarks(results.left_hand_landmarks.landmark, (480, 640)) if results.left_hand_landmarks else np.zeros(21*3)
    rh = normalize_landmarks(results.right_hand_landmarks.landmark, (480, 640)) if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

def augment_data(sequence):
    augmented_sequence = []
    for frame in sequence:
        augmented_frame = frame.copy()
        # Flip x-coordinates: For each set of (x, y, z), flip the x value
        augmented_frame[::3] = -augmented_frame[::3]  # Flip every third value starting from index 0 (x-coordinates)
        augmented_sequence.append(augmented_frame)
    return augmented_sequence

# Path for the exported data (numpy arrays)
DATA_PATH = os.path.join('MP_Data')

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Actions we want to detect
actions = np.array(['busy', 'deaf', 'excuse me', 'fine', 'good', 'goodbye', 'hard of hearing', 'hearing', 'hello', 'help', 'how', 'please', 'thank you'])

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
        if os.path.isdir(sequence_path):
            for frame_file in sorted(os.listdir(sequence_path), key=lambda x: int(x.split('.')[0])):
                frame_path = os.path.join(sequence_path, frame_file)
                res = np.load(frame_path)
                window.append(res)

        sequences.append(window)
        labels.append(label_map[action])

        # Apply augmentation
        augmented_seq = augment_data(window)
        sequences.append(augmented_seq)
        labels.append(label_map[action])

# Set up data for training
x = np.array(sequences)
y = to_categorical(labels).astype(int)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

# Model creation
model = Sequential()
model.add(Bidirectional(GRU(128, return_sequences=True, activation='relu', input_shape=(x_train.shape[0], x_train.shape[1]))))
model.add(Dropout(0.3))
model.add(GRU(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
model.add(Dense(actions.shape[0], activation='softmax'))

# Use RMSprop optimizer with a lower learning rate
optimizer = RMSprop(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=170, validation_split=0.1, callbacks=[tb_callback, early_stopping, reduce_lr])

# Evaluate the model
yhat = model.predict(x_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

# Display evaluation metrics
print(multilabel_confusion_matrix(ytrue, yhat))
print(f'Accuracy: {accuracy_score(ytrue, yhat)}')

# Save the trained model
#model.save('model.h5')
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)