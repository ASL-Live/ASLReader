import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5", "model/labels.txt")

offset = 20
imgSize = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U","V", "W", "X", "Y"]

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            imgCropResized = cv2.resize(imgCrop, (imgSize, imgSize))
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgWhite[0:imgCropResized.shape[0], 0:imgCropResized.shape[1]] = imgCropResized

            # Prediction
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

            # Display the letter and confidence
            if index < len(labels):
                letter = labels[index]
                confidence = prediction[index]
                text = f'{letter}, {confidence:.2f}'
                cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow('imageWhite', imgWhite)

    cv2.imshow('image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
