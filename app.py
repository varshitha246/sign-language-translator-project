import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image

# Backend URL (Replace with your backend API endpoint)
BACKEND_URL = "http://127.0.0.1:5000/predict"  # Update this with the correct backend URL

st.set_page_config(page_title="Sign Language Translator", layout="centered")

st.title("🖐 Sign Language Translator")
st.write("Translate Sign Language into Text and Speech")

# Upload Image or Capture from Webcam
option = st.radio("Choose an option:", ("Upload an Image", "Use Webcam"))

def process_image(image):
    img = Image.open(image)
    img = img.convert("RGB")
    return img

def capture_webcam():
    cap = cv2.VideoCapture(0)
    st.write("Capturing from webcam...")
    ret, frame = cap.read()
    cap.release()
    if ret:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        return None

if option == "Upload an Image":
    uploaded_file = st.file_uploader("Upload an image of sign language", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = process_image(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Send to Backend for Prediction
        with st.spinner("Translating..."):
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(BACKEND_URL, files=files)
            if response.status_code == 200:
                result = response.json()
                translation_map = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
                                   'a': 'a', 'b': 'b', 'c': 'c', 'd': 'd', 'e': 'e', 'f': 'f', 'g': 'g', 'h': 'h', 'i': 'i', 'j': 'j',
                                   'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'o': 'o', 'p': 'p', 'q': 'q', 'r': 'r', 's': 's', 't': 't',
                                   'u': 'u', 'v': 'v', 'w': 'w', 'x': 'x', 'y': 'y', 'z': 'z'}
                translation_text = translation_map.get(result['translation'], "Unknown")
                st.success(f"Translation: {translation_text}")
            else:
                st.error("Error in translation! Try again.")

elif option == "Use Webcam":
    if st.button("Capture Image from Webcam"):
        img = capture_webcam()
        if img is not None:
            st.image(img, caption="Captured Image", use_column_width=True)
            img_bytes = np.array(img)
            _, img_encoded = cv2.imencode(".jpg", img_bytes)
            img_bytes = img_encoded.tobytes()
            
            # Send to Backend for Prediction
            with st.spinner("Translating..."):
                files = {"file": img_bytes}
                response = requests.post(BACKEND_URL, files=files)
                if response.status_code == 200:
                    result = response.json()
                    translation_map = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
                                       'a': 'a', 'b': 'b', 'c': 'c', 'd': 'd', 'e': 'e', 'f': 'f', 'g': 'g', 'h': 'h', 'i': 'i', 'j': 'j',
                                       'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'o': 'o', 'p': 'p', 'q': 'q', 'r': 'r', 's': 's', 't': 't',
                                       'u': 'u', 'v': 'v', 'w': 'w', 'x': 'x', 'y': 'y', 'z': 'z'}
                    translation_text = translation_map.get(result['translation'], "Unknown")
                    st.success(f"Translation: {translation_text}")
                else:
                    st.error("Error in translation! Try again.")

st.markdown("---")
st.write("Developed by Your Name")



