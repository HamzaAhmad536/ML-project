import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from keras import layers, models
import os
import pygame
import threading

# Initialize pygame mixer for alarm
pygame.mixer.init()
ALARM_FILE = 'alarm.wav'

# Initialize MediaPipe Face Mesh with improved parameters
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,  # Further lowered for better detection
    min_tracking_confidence=0.3    # Further lowered for better detection
)

# Define the eye landmarks indices
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

def enhance_eye_image(eye_img):
    """Enhance eye image for better feature extraction"""
    # Convert to grayscale
    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
    
    return thresh

def preprocess_eye_image(eye_img):
    """Preprocess eye image for model input"""
    # Enhance the image
    enhanced = enhance_eye_image(eye_img)
    
    # Resize to 64x64
    resized = cv2.resize(enhanced, (64, 64))
    
    # Normalize
    normalized = resized / 255.0
    
    # Add channel dimension
    return np.expand_dims(normalized, axis=-1)

def get_eye_region(frame, landmarks, eye_indices):
    """Extract eye region with improved accuracy"""
    h, w = frame.shape[:2]
    
    # Get all landmark coordinates
    x_coords = [landmarks.landmark[idx].x * w for idx in eye_indices]
    y_coords = [landmarks.landmark[idx].y * h for idx in eye_indices]
    
    # Calculate eye region with padding
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    
    # Add dynamic padding based on eye size
    eye_width = x_max - x_min
    eye_height = y_max - y_min
    padding_x = int(eye_width * 0.3)  # 30% padding
    padding_y = int(eye_height * 0.3)
    
    # Ensure padding doesn't exceed image boundaries
    x_min = max(0, x_min - padding_x)
    x_max = min(w, x_max + padding_x)
    y_min = max(0, y_min - padding_y)
    y_max = min(h, y_max + padding_y)
    
    # Extract eye region
    eye_region = frame[y_min:y_max, x_min:x_max]
    
    # Ensure minimum size
    if eye_region.size == 0 or eye_region.shape[0] < 10 or eye_region.shape[1] < 10:
        return None
    
    return eye_region

def calculate_eye_aspect_ratio(landmarks, eye_indices):
    """Calculate eye aspect ratio for additional validation"""
    # Get vertical landmarks
    top = landmarks.landmark[eye_indices[1]].y
    bottom = landmarks.landmark[eye_indices[5]].y
    
    # Get horizontal landmarks
    left = landmarks.landmark[eye_indices[0]].x
    right = landmarks.landmark[eye_indices[3]].x
    
    # Calculate aspect ratio
    height = abs(top - bottom)
    width = abs(right - left)
    
    if width == 0:
        return 0
    
    return height / width

def train_and_save_model():
    print("Training new model...")
    model = create_model()
    
    # Load and preprocess training data
    train_dir = 'MRL_EYE/train'
    X_train = []
    y_train = []
    
    # Load awake (open) eyes
    awake_dir = os.path.join(train_dir, 'awake')
    for img_name in os.listdir(awake_dir):
        img_path = os.path.join(awake_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            X_train.append(img)
            y_train.append(1)  # 1 for open/awake
    
    # Load sleepy (closed) eyes
    sleepy_dir = os.path.join(train_dir, 'sleepy')
    for img_name in os.listdir(sleepy_dir):
        img_path = os.path.join(sleepy_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            X_train.append(img)
            y_train.append(0)  # 0 for closed/sleepy
    
    X_train = np.array(X_train) / 255.0
    X_train = np.expand_dims(X_train, axis=-1)
    y_train = np.array(y_train)
    
    # Train the model with more epochs and early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    model.fit(X_train, y_train, 
              epochs=6,  # Increased epochs
              batch_size=32, 
              validation_split=0.2,
              callbacks=[early_stopping])
    
    # Save the model
    model.save('eye_state_model.h5')
    print("Model saved as 'eye_state_model.h5'")
    return model

def load_or_train_model():
    try:
        if os.path.exists('eye_state_model.h5'):
            print("Loading existing model...")
            model = tf.keras.models.load_model('eye_state_model.h5')
            print("Model loaded successfully")
            return model
        else:
            print("No existing model found, training new model...")
            return train_and_save_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Training new model...")
        return train_and_save_model()

def play_alarm():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load(ALARM_FILE)
        pygame.mixer.music.play(-1)  # Loop alarm

def stop_alarm():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

def main():
    model = load_or_train_model()
    # Use only laptop camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {frame_width}x{frame_height}")
    prediction_history = []
    history_size = 5
    frame_count = 0
    process_every_n_frames = 1
    closed_frames = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 20  # fallback default
    closed_threshold = int(fps * 2)  # 2 seconds
    alarm_playing = False
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break
        frame_count += 1
        if frame_count % process_every_n_frames != 0:
            continue
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        eyes_closed = False
        eyes_detected = False
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye = get_eye_region(frame, face_landmarks, LEFT_EYE_INDICES)
                right_eye = get_eye_region(frame, face_landmarks, RIGHT_EYE_INDICES)
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                if left_eye is not None and right_eye is not None:
                    eyes_detected = True
                    try:
                        left_ear = calculate_eye_aspect_ratio(face_landmarks, LEFT_EYE_INDICES)
                        right_ear = calculate_eye_aspect_ratio(face_landmarks, RIGHT_EYE_INDICES)
                        left_eye_processed = preprocess_eye_image(left_eye)
                        right_eye_processed = preprocess_eye_image(right_eye)
                        left_pred = model.predict(np.expand_dims(left_eye_processed, axis=0), verbose=0)[0][0]
                        right_pred = model.predict(np.expand_dims(right_eye_processed, axis=0), verbose=0)[0][0]
                        avg_pred = (left_pred + right_pred) / 2
                        avg_ear = (left_ear + right_ear) / 2
                        prediction_history.append(avg_pred)
                        if len(prediction_history) > history_size:
                            prediction_history.pop(0)
                        smoothed_pred = sum(prediction_history) / len(prediction_history)
                        eye_state = "Open" if smoothed_pred > 0.5 and avg_ear > 0.15 else "Closed"
                        confidence = smoothed_pred if smoothed_pred > 0.5 else 1 - smoothed_pred
                        cv2.putText(frame, f"Eyes: {eye_state} ({confidence:.2f})", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 90),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        color = (0, 255, 0) if eye_state == "Open" else (0, 0, 255)
                        cv2.rectangle(frame, 
                                    (int(face_landmarks.landmark[LEFT_EYE_INDICES[0]].x * frame.shape[1]),
                                     int(face_landmarks.landmark[LEFT_EYE_INDICES[0]].y * frame.shape[0])),
                                    (int(face_landmarks.landmark[LEFT_EYE_INDICES[8]].x * frame.shape[1]),
                                     int(face_landmarks.landmark[LEFT_EYE_INDICES[8]].y * frame.shape[0])),
                                    color, 2)
                        cv2.rectangle(frame,
                                    (int(face_landmarks.landmark[RIGHT_EYE_INDICES[0]].x * frame.shape[1]),
                                     int(face_landmarks.landmark[RIGHT_EYE_INDICES[0]].y * frame.shape[0])),
                                    (int(face_landmarks.landmark[RIGHT_EYE_INDICES[8]].x * frame.shape[1]),
                                     int(face_landmarks.landmark[RIGHT_EYE_INDICES[8]].y * frame.shape[0])),
                                    color, 2)
                        if eye_state == "Closed":
                            eyes_closed = True
                        else:
                            eyes_closed = False
                    except Exception as e:
                        print(f"Error processing eyes: {e}")
                        cv2.putText(frame, "Error processing eyes", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "No eyes detected", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Alarm logic (trigger if eyes closed OR no eyes detected for >2s)
        if (eyes_detected and eyes_closed) or (not eyes_detected):
            closed_frames += 1
            if closed_frames > closed_threshold:
                play_alarm()
        else:
            closed_frames = 0
            stop_alarm()
        cv2.imshow('Eye State Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    stop_alarm()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 