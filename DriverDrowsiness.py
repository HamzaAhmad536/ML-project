import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from keras import models
import os
import pygame
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

# Landmark indices for eyes and mouth
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
MOUTH_INNER_INDICES = [13, 14]

# Configuration
EYE_MODEL_PATH = 'eye_state_model.h5'
EYE_IMG_SIZE = (64, 64)
EYE_STATE_THRESHOLD = 0.5
EAR_THRESHOLD = 0.15
CLOSED_FRAMES_THRESHOLD_SECONDS = 0.5
YAWN_DURATION_THRESHOLD_SECONDS = 1.5
YAWN_LIP_DISTANCE_THRESHOLD = 25
PREDICTION_HISTORY_SIZE = 5

# Initialize alarm system
alarm_sound_available = False
alarm_playing = False

try:
    pygame.mixer.init()
    if os.path.exists('alarm.wav'):
        pygame.mixer.music.load('alarm.wav')
        alarm_sound_available = True
except pygame.error as e:
    print(f"Audio initialization error: {e}")

def play_alarm():
    global alarm_playing
    if alarm_sound_available and not alarm_playing:
        pygame.mixer.music.play(-1)
        alarm_playing = True

def stop_alarm():
    global alarm_playing
    if alarm_playing:
        pygame.mixer.music.stop()
        alarm_playing = False

def enhance_eye_image(eye_img):
    if eye_img is None or eye_img.size == 0:
        return None
    
    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY) if len(eye_img.shape) == 3 else eye_img
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return cv2.GaussianBlur(clahe.apply(gray), (5, 5), 0)

def preprocess_eye_image(eye_img):
    enhanced = enhance_eye_image(eye_img)
    if enhanced is None:
        return None
    resized = cv2.resize(enhanced, EYE_IMG_SIZE)
    return np.expand_dims(resized / 255.0, axis=-1)

def get_eye_region(frame, landmarks, eye_indices):
    if frame is None or landmarks is None:
        return None
    
    h, w = frame.shape[:2]
    x_coords = [landmarks.landmark[idx].x * w for idx in eye_indices]
    y_coords = [landmarks.landmark[idx].y * h for idx in eye_indices]

    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))

    padding_x = int((x_max - x_min) * 0.3)
    padding_y = int((y_max - y_min) * 0.3)

    x_min = max(0, x_min - padding_x)
    x_max = min(w, x_max + padding_x)
    y_min = max(0, y_min - padding_y)
    y_max = min(h, y_max + padding_y)

    eye_region = frame[y_min:y_max, x_min:x_max]
    return eye_region if eye_region.size > 0 else None

def calculate_eye_aspect_ratio(landmarks, eye_indices):
    try:
        y_top = landmarks.landmark[eye_indices[12]].y
        y_bottom = landmarks.landmark[eye_indices[4]].y
        x_left = landmarks.landmark[eye_indices[0]].x
        x_right = landmarks.landmark[eye_indices[8]].x
        
        height = abs(y_top - y_bottom)
        width = abs(x_right - x_left)
        
        return (height / width) * 100 if width != 0 else 0
    except Exception:
        return 0

def load_model():
    if os.path.exists(EYE_MODEL_PATH):
        return tf.keras.models.load_model(EYE_MODEL_PATH)
    return None

def main():
    eye_model = load_model()
    if eye_model is None:
        print("Error: Failed to load eye state model")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    closed_frames_threshold = int(CLOSED_FRAMES_THRESHOLD_SECONDS * fps)

    # State tracking variables
    prediction_history = []
    closed_frames_counter = 0
    yawn_start_time = 0
    eye_closed_start_time = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        current_time = time.time()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Face detection
        results = face_mesh.process(rgb_frame)
        face_detected = bool(results.multi_face_landmarks)
        
        # Reset frame states
        eyes_closed = False
        yawning = False
        ear_value = 0
        lip_distance = 0

        if face_detected:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Eye state detection
            left_eye = get_eye_region(frame, face_landmarks, LEFT_EYE_INDICES)
            right_eye = get_eye_region(frame, face_landmarks, RIGHT_EYE_INDICES)
            
            if left_eye is not None and right_eye is not None:
                left_ear = calculate_eye_aspect_ratio(face_landmarks, LEFT_EYE_INDICES)
                right_ear = calculate_eye_aspect_ratio(face_landmarks, RIGHT_EYE_INDICES)
                ear_value = (left_ear + right_ear) / 2.0

                left_processed = preprocess_eye_image(left_eye)
                right_processed = preprocess_eye_image(right_eye)
                
                if left_processed is not None and right_processed is not None:
                    left_pred = eye_model.predict(np.expand_dims(left_processed, axis=0), verbose=0)[0][0]
                    right_pred = eye_model.predict(np.expand_dims(right_processed, axis=0), verbose=0)[0][0]
                    eye_pred = (left_pred + right_pred) / 2.0
                    
                    prediction_history.append(eye_pred)
                    if len(prediction_history) > PREDICTION_HISTORY_SIZE:
                        prediction_history.pop(0)
                    
                    smoothed_pred = sum(prediction_history) / len(prediction_history)
                    eyes_closed = smoothed_pred < EYE_STATE_THRESHOLD or ear_value < EAR_THRESHOLD
                    
                    status_text = f"{'Closed' if eyes_closed else 'Open'} (P:{smoothed_pred:.2f} E:{ear_value:.1f})"
                    cv2.putText(frame, f"Eyes: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if eyes_closed else (0, 255, 0), 2)
                    
                    # Update eye closed timer
                    if eyes_closed and eye_closed_start_time == 0:
                        eye_closed_start_time = current_time
                    elif not eyes_closed and eye_closed_start_time != 0:
                        eye_closed_start_time = 0

            # Yawn detection
            try:
                top_lip = face_landmarks.landmark[MOUTH_INNER_INDICES[0]]
                bottom_lip = face_landmarks.landmark[MOUTH_INNER_INDICES[1]]
                lip_distance = abs(top_lip.y - bottom_lip.y) * frame.shape[0]
                
                cv2.putText(frame, f"Mouth Open: {lip_distance:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                yawning = lip_distance > YAWN_LIP_DISTANCE_THRESHOLD
                if yawning and yawn_start_time == 0:
                    yawn_start_time = current_time
                elif not yawning and yawn_start_time != 0:
                    yawn_start_time = 0
                    
            except Exception:
                yawn_start_time = 0

            # Display timers
            if eye_closed_start_time != 0:
                closed_duration = current_time - eye_closed_start_time
                cv2.putText(frame, f"Closed Time: {closed_duration:.1f}s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if yawn_start_time != 0:
                yawn_duration = current_time - yawn_start_time
                cv2.putText(frame, f"Yawn Time: {yawn_duration:.1f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Drowsiness detection logic
        if eyes_closed:
            closed_frames_counter += 1
            if closed_frames_counter >= closed_frames_threshold:
                cv2.putText(frame, "ALERT: EYES CLOSED", (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                play_alarm()
        else:
            closed_frames_counter = 0
            stop_alarm()

        if yawn_start_time != 0 and (current_time - yawn_start_time) >= YAWN_DURATION_THRESHOLD_SECONDS:
            cv2.putText(frame, "ALERT: PROLONGED YAWN", (10, frame.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            play_alarm()

        cv2.imshow('Drowsiness Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    stop_alarm()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()