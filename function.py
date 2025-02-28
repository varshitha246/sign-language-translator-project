import cv2
import numpy as np
import os
import random
import mediapipe as mp

# Initialize Mediapipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Define dataset path
DATA_PATH = "MP_Data"

# Define actions (letters & numbers) - Adjust according to your dataset
actions = np.array([
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 
    'u', 'v', 'w', 'x', 'y', 'z'
])

# Number of sequences per action
no_sequences = 70  # Adjust based on dataset

# Length of each sequence (number of frames per sequence)
sequence_length = 30  

# Function to create dataset directories
def create_dataset_folders():
    """Creates dataset directories for storing keypoints if they don't already exist."""
    for action in actions:
        for sequence in range(1, no_sequences + 1):  # Ensure correct sequence numbering
            dir_path = os.path.join(DATA_PATH, action, str(sequence))
            os.makedirs(dir_path, exist_ok=True)

# Call the function to create directories
create_dataset_folders()
print("Dataset folders verified/created successfully.")

# Function for processing images with Mediapipe
def mediapipe_detection(image, model):
    """Processes an image using Mediapipe Hands model and returns the results."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    image.flags.writeable = False  
    results = model.process(image)  
    image.flags.writeable = True  
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  
    return image, results

# Function to draw landmarks on the image
def draw_styled_landmarks(image, results):
    """Draws hand landmarks with Mediapipe styles."""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS, 
                mp_drawing_styles.get_default_hand_landmarks_style(), 
                mp_drawing_styles.get_default_hand_connections_style()
            )

# Augmentation Function
def augment_keypoints(keypoints):
    """Applies random augmentation to keypoints to prevent overfitting."""
    if random.random() > 0.5:  
        keypoints = keypoints + np.random.normal(0, 0.02, keypoints.shape)  
    return keypoints

# Normalize Keypoints
def normalize_keypoints(keypoints):
    """Normalizes keypoints between -1 and 1."""
    return (keypoints - np.min(keypoints)) / (np.max(keypoints) - np.min(keypoints) + 1e-6)  

# Function to extract keypoints from hand landmarks
def extract_keypoints(results):
    """Extracts keypoints from detected hand landmarks. Returns zero array if no hands detected."""
    
    keypoints = np.zeros(21 * 3)  # Single hand → 21 keypoints (x, y, z) → 63 values

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]  # Only first detected hand
        keypoints = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
        keypoints = normalize_keypoints(keypoints)  
        keypoints = augment_keypoints(keypoints)  
    
    return keypoints  # Shape: (63,)


