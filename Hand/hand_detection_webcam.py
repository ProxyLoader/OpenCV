import string

import cv2
import mediapipe
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="./hand_landmarker.task"),
    running_mode=vision.RunningMode.IMAGE
)

detector = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
names = [
    "WRIST",
    "THUMB_CMC","THUMB_MCP","THUMB_IP","THUMB_TIP",
    "INDEX_MCP","INDEX_PIP","INDEX_DIP","INDEX_TIP",
    "MIDDLE_MCP","MIDDLE_PIP","MIDDLE_DIP","MIDDLE_TIP",
    "RING_MCP","RING_PIP","RING_DIP","RING_TIP",
    "PINKY_MCP","PINKY_PIP","PINKY_DIP","PINKY_TIP"
]

while True:
    ret, frame = cap.read()
    img = frame.copy()

    if img is None:
        print("Image not found")
        break

    h, w, _ = img.shape

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            for i,lm in enumerate(hand):
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.putText(
                    img,
                    names[i],
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 0, 0),
                    1,)

                cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
    cv2.imshow("img", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
