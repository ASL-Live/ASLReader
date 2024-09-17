import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2, l1_l2
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Load global mean and std
global_mean = np.load('global_mean.npy')
global_std = np.load('global_std.npy')


def normalize_keypoints(keypoints):
    return (keypoints - global_mean) / global_std


# Path for the exported data (numpy arrays)
DATA_PATH = os.path.join('MP_Data')

# Actions we want to detect
actions = np.array(['dad', 'food', 'tomorrow'])

label_map = {label: num for num, label in enumerate(actions)}

# Sequence length
SEQUENCE_LENGTH = 30  # Ensure this matches your data preprocessing step

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

        # Get all frame files and sort them
        frame_files = sorted(os.listdir(sequence_path), key=lambda x: int(x.split('.')[0]))

        # Ensure that we have exactly SEQUENCE_LENGTH frames
        if len(frame_files) != SEQUENCE_LENGTH:
            # If more frames, truncate
            if len(frame_files) > SEQUENCE_LENGTH:
                frame_files = frame_files[:SEQUENCE_LENGTH]
            # If fewer frames, pad by duplicating the last frame
            else:
                last_frame = frame_files[-1]
                while len(frame_files) < SEQUENCE_LENGTH:
                    frame_files.append(last_frame)

        # Iterate over the frame files
        for frame_file in frame_files:
            frame_path = os.path.join(sequence_path, frame_file)
            keypoints = np.load(frame_path)
            normalized_keypoints = normalize_keypoints(keypoints)
            window.append(normalized_keypoints)

        sequences.append(window)
        labels.append(label_map[action])

        # Apply augmentation (optional)
        # For example, flip keypoints horizontally
        augmented_window = []
        for keypoints in window:
            augmented_keypoints = keypoints.copy()
            augmented_keypoints[::3] = -augmented_keypoints[::3]  # Flip x-coordinates
            augmented_window.append(augmented_keypoints)
        sequences.append(augmented_window)
        labels.append(label_map[action])

# Convert sequences and labels to numpy arrays
x = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, shuffle=True)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Model creation
model = Sequential()
model.add(Bidirectional(GRU(128, return_sequences=True, activation='relu',
                            input_shape=(SEQUENCE_LENGTH, x_train.shape[2]))))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(GRU(64, return_sequences=False, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.02, l2=0.02)))  # Slightly increased regularization
model.add(Dense(actions.shape[0], activation='softmax'))

# Use Adam optimizer with a lower learning rate
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=40, validation_split=0.1,
                    callbacks=[tb_callback, early_stopping, reduce_lr])

# Evaluate the model
yhat = model.predict(x_test)
ytrue = np.argmax(y_test, axis=1)
yhat = np.argmax(yhat, axis=1)

# Display evaluation metrics
print(multilabel_confusion_matrix(ytrue, yhat))
print(f'Accuracy: {accuracy_score(ytrue, yhat)}')

# Save the trained model
model.save('model.h5')

# Plot accuracy
plt.figure(figsize=(8, 4))
plt.plot(history.history['categorical_accuracy'], label='Training Accuracy')
plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Plot loss
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
