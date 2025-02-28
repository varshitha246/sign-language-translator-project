import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# 🏗️ Load Trained Model
model_path = "best_model.h5"
try:
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f" Error loading model: {e}")
    exit()

# 🎯 Define Class Labels (Update if needed)
actions = np.array([
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z'
])

# 🎥 Initialize Webcam
cap = cv2.VideoCapture(0)

# 📌 Preprocessing Function
def preprocess_image(frame):
    frame_resized = cv2.resize(frame, (224, 224))  # ✅ Fix: Resize to 224x224
    frame_normalized = frame_resized / 255.0
    return np.expand_dims(frame_normalized, axis=0)  # Shape: (1, 224, 224, 3)

# 🔍 Video Capture Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 📌 Define Region of Interest (ROI) for Hand Detection
    x, y, w, h = 100, 100, 300, 300  # Adjust as needed
    roi = frame[y:y+h, x:x+w]

    # 🎯 Process Image
    processed_input = preprocess_image(roi)

    # 🔮 Predict
    prediction = model.predict(processed_input)
    predicted_class = np.argmax(prediction)
    predicted_action = actions[predicted_class]

    # 🖍️ Draw Bounding Box & Prediction
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    text = f"Output: {predicted_action}"
    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 🎥 Show Webcam Feed
    cv2.imshow("Sign Language Recognition", frame)

    # ❌ Press 'q' to Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 📌 Release Resources
cap.release()
cv2.destroyAllWindows()









    