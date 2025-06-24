import os
import onnxruntime as ort
import cv2
import mediapipe as mp
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

session = ort.InferenceSession('sign_mnist.onnx')
input_name = session.get_inputs()[0].name

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
h, w, _ = frame.shape

letterpred = [
    'A','B','C','D','E','F','G','H','I','K',
    'L','M','N','O','P','Q','R','S','T','U',
    'V','W','X','Y'
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks

    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max, y_max = 0, 0
            x_min, y_min = w, h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y

            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            roi_color = frame[y_min:y_max, x_min:x_max]
            roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
            roi_resized = cv2.resize(roi_gray, (28, 28))

            pixeldata = roi_resized.astype(np.float32) / 255.0
            pixeldata = pixeldata.reshape(1, 28, 28, 1)

            prediction = session.run(None, {input_name: pixeldata})[0][0]
            top5_idx = np.argsort(prediction)[-5:][::-1]

            for i, idx in enumerate(top5_idx):
                letter = letterpred[idx]
                conf = prediction[idx] * 100
                text = f"{i+1}: {letter} {conf:.1f}%"
                text_x = x_min + 5
                text_y = y_min + 25 + i * 25
                cv2.putText(
                    frame,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            break

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
