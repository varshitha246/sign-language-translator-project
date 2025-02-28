import os
import cv2
import numpy as np
import mediapipe as mp

# Define dataset path
DATA_PATH = r"C:\Users\varsh\Desktop\project slp\Dataset"

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Define function to extract keypoints from an image
def extract_keypoints(image):
    with mp_holistic.Holistic(static_image_mode=True) as holistic:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        
        # Extract keypoints from results
        pose = results.pose_landmarks.landmark if results.pose_landmarks else []
        face = results.face_landmarks.landmark if results.face_landmarks else []
        left_hand = results.left_hand_landmarks.landmark if results.left_hand_landmarks else []
        right_hand = results.right_hand_landmarks.landmark if results.right_hand_landmarks else []

        # Convert keypoints to a flat array
        pose_keypoints = np.array([[p.x, p.y, p.z] for p in pose]).flatten() if pose else np.zeros(33 * 3)
        face_keypoints = np.array([[f.x, f.y, f.z] for f in face]).flatten() if face else np.zeros(468 * 3)
        left_hand_keypoints = np.array([[lh.x, lh.y, lh.z] for lh in left_hand]).flatten() if left_hand else np.zeros(21 * 3)
        right_hand_keypoints = np.array([[rh.x, rh.y, rh.z] for rh in right_hand]).flatten() if right_hand else np.zeros(21 * 3)

        # Concatenate all keypoints into a single array
        return np.concatenate([pose_keypoints, face_keypoints, left_hand_keypoints, right_hand_keypoints])

# Process each image in the dataset and save keypoints
for action in os.listdir(DATA_PATH):
    action_path = os.path.join(DATA_PATH, action)
    
    if os.path.isdir(action_path):  # Ensure it's a directory
        for img_file in os.listdir(action_path):
            if img_file.endswith('.jpeg') or img_file.endswith('.jpg') or img_file.endswith('.png'):
                img_path = os.path.join(action_path, img_file)
                image = cv2.imread(img_path)

                if image is None:
                    print(f"Skipping {img_path} (cannot read)")
                    continue

                keypoints = extract_keypoints(image)
                
                # Define save path
                save_path = img_path.replace('.jpeg', '.npy').replace('.jpg', '.npy').replace('.png', '.npy')
                np.save(save_path, keypoints)

                print(f"Saved keypoints: {save_path}")

print("Keypoint extraction completed.")
