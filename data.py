import os
import cv2
import numpy as np
import mediapipe as mp
from function import mediapipe_detection, draw_styled_landmarks, extract_keypoints
from time import sleep

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands

# Define dataset path
DATA_PATH = os.path.abspath(r"C:\Users\varsh\Desktop\project slp\Dataset")

# Define 36 actions (Numbers 0-9 + Alphabet a-z)
actions = [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('z') + 1)]

# Optimized parameters
no_sequences = 35  # Reduced from 70 → Prevents redundancy
sequence_length = 10  # Reduced from 30 → Prevents overfitting

# Ensure dataset directories exist
for action in actions:
    for sequence in range(1, no_sequences + 1):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

# Process images with Mediapipe
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    for action in actions:
        print(f" Collecting data for action: {action}")
        
        for sequence in range(1, no_sequences + 1):  
            print(f"Sequence {sequence}/{no_sequences}")
            
            for frame_num in range(1, sequence_length + 1):  
                image_path = os.path.join(DATA_PATH, action, f"{action}_({frame_num}).jpeg")
                frame = cv2.imread(image_path)

                if frame is None:
                    print(f" Error: Unable to read {image_path}. Skipping.")
                    continue
                
                # Perform hand landmark detection
                image, results = mediapipe_detection(frame, hands)
                draw_styled_landmarks(image, results)
                
                # Display "STARTING COLLECTION" message for the first frame
                if frame_num == 1:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    sleep(0.5)  

                # Display frame collection status
                cv2.putText(image, f'Collecting: {action} | Video {sequence} | Frame {frame_num}', 
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                
                # Show image
                cv2.imshow('OpenCV Feed', image)
                
                # Extract keypoints and save them
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                np.save(npy_path, keypoints)

                # Press 'q' to exit early
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print(" Collection interrupted by user.")
                    cv2.destroyAllWindows()
                    exit()

cv2.destroyAllWindows()
print("Data collection completed successfully!")

































