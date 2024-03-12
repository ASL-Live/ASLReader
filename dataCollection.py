import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300
counter = 0

folder = 'data/Y'

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Check if imgCrop dimensions are valid
        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            cv2.imshow('imageCrop', imgCrop)

            # Resize cropped image to match the size of the white image
            imgCropResized = cv2.resize(imgCrop, (imgSize, imgSize))

            # Create a white background image
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            # Overlay the cropped image onto the white image
            imgWhite[0:imgCropResized.shape[0], 0:imgCropResized.shape[1]] = imgCropResized

            # Display the overlaid image
            cv2.imshow('imageWhite', imgWhite)

    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    key = cv2.waitKey(1)
    if key == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)


cap.release()
cv2.destroyAllWindows()
