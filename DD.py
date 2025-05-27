# Combined Drowsiness Detection (Eye State + Yawn)

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from keras import layers, models
import os
import pygame
import time
import threading # Kept import in case needed later, though current logic doesn't use explicit threading

# --- Configuration --- #

# Alarm Settings
ALARM_FILE = 'alarm.wav' # Ensure this file exists in the same directory

# MediaPipe Face Mesh Initialization (Using refined parameters from eye detection)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.4, # Adjusted confidence slightly
    min_tracking_confidence=0.4
)

# Eye Landmark Indices (from eye detection)
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Mouth Landmark Indices (from MediaPipe documentation for inner lips)
MOUTH_INNER_INDICES = [13, 14] # Top lip: 13, Bottom lip: 14

# Eye State Model Configuration
EYE_MODEL_PATH = 'eye_state_model.h5'
EYE_IMG_SIZE = (64, 64)
EYE_STATE_THRESHOLD = 0.5 # Model prediction threshold for 'Open'
EAR_THRESHOLD = 0.15      # Eye Aspect Ratio threshold

# Drowsiness Thresholds
CLOSED_FRAMES_THRESHOLD_SECONDS = 0.5 # Eyes closed/undetected duration to trigger alarm
YAWN_DURATION_THRESHOLD_SECONDS = 1.5 # Yawn duration to trigger alarm
YAWN_LIP_DISTANCE_THRESHOLD = 25     # Vertical lip distance threshold for yawn detection (adjust as needed)

# Frame Processing Settings
PROCESS_EVERY_N_FRAMES = 1 # Process every frame
PREDICTION_HISTORY_SIZE = 5 # Smoothing window for eye state prediction

# --- Pygame Initialization ---
try:
    pygame.mixer.init()
    # Check if alarm file exists
    if not os.path.exists(ALARM_FILE):
        print(f"Warning: Alarm file '{ALARM_FILE}' not found. Alarm will not sound.")
        alarm_sound_available = False
    else:
        # Use pygame.mixer.music for looping capability
        pygame.mixer.music.load(ALARM_FILE)
        alarm_sound_available = True
except pygame.error as e:
    print(f"Error initializing pygame mixer: {e}. Alarm functionality disabled.")
    alarm_sound_available = False

alarm_playing = False

# --- Alarm Control Functions ---
def play_alarm():
    global alarm_playing
    if alarm_sound_available and not alarm_playing:
        try:
            pygame.mixer.music.play(-1) # Loop indefinitely
            alarm_playing = True
            print("Alarm Started")
        except pygame.error as e:
            print(f"Error playing alarm: {e}")

def stop_alarm():
    global alarm_playing
    if alarm_sound_available and alarm_playing:
        try:
            pygame.mixer.music.stop()
            alarm_playing = False
            print("Alarm Stopped")
        except pygame.error as e:
            print(f"Error stopping alarm: {e}")

def enhance_eye_image(eye_img):
    if eye_img is None or eye_img.size == 0:
        return None
    if len(eye_img.shape) == 3 and eye_img.shape[2] == 3:
        gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    elif len(eye_img.shape) == 2:
        gray = eye_img # Already grayscale
    else:
        print("Warning: Unexpected eye image format for enhancement.")
        return None

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    # No thresholding here, keep grayscale for model
    return blurred

def preprocess_eye_image(eye_img):
    if eye_img is None or eye_img.size == 0:
        return None
    enhanced = enhance_eye_image(eye_img)
    if enhanced is None:
        return None
    resized = cv2.resize(enhanced, EYE_IMG_SIZE)
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=-1) # Add channel dimension

def get_eye_region(frame, landmarks, eye_indices):
    if frame is None or landmarks is None:
        return None
    h, w = frame.shape[:2]
    try:
        x_coords = [landmarks.landmark[idx].x * w for idx in eye_indices]
        y_coords = [landmarks.landmark[idx].y * h for idx in eye_indices]

        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        eye_width = x_max - x_min
        eye_height = y_max - y_min
        padding_x = int(eye_width * 0.3)
        padding_y = int(eye_height * 0.3)

        x_min = max(0, x_min - padding_x)
        x_max = min(w, x_max + padding_x)
        y_min = max(0, y_min - padding_y)
        y_max = min(h, y_max + padding_y)

        eye_region = frame[y_min:y_max, x_min:x_max]

        if eye_region.size == 0 or eye_region.shape[0] < 5 or eye_region.shape[1] < 5:
            return None
        return eye_region
    except (IndexError, ValueError, TypeError) as e:
        print(f"Error extracting eye region: {e}")
        return None

def calculate_eye_aspect_ratio(landmarks, eye_indices):
    try:
        # Points based on MediaPipe landmark indices for vertical distance
        # Example: Using points around the pupil vertical axis
        y_top = landmarks.landmark[eye_indices[12]].y # Approx upper eyelid center
        y_bottom = landmarks.landmark[eye_indices[4]].y # Approx lower eyelid center

        # Points for horizontal distance
        x_left = landmarks.landmark[eye_indices[0]].x # Left corner
        x_right = landmarks.landmark[eye_indices[8]].x # Right corner

        height = abs(y_top - y_bottom)
        width = abs(x_right - x_left)

        if width == 0:
            return 0
        ear = height / width
        # Scale EAR based on typical values if needed, mediapipe coords are normalized
        # EAR calculation might need tuning based on specific landmark choices
        return ear * 100 # Scale for better readability/thresholding
    except (IndexError, AttributeError) as e:
        print(f"Error calculating EAR: {e}")
        return 0

def load_model():

        if os.path.exists(EYE_MODEL_PATH):
            print(f"Loading existing eye state model from {EYE_MODEL_PATH}...")
            # Load model with custom objects if any (e.g., custom layers/metrics)
            # For standard layers, this should work:
            model = tf.keras.models.load_model(EYE_MODEL_PATH)
            print("Eye state model loaded successfully.")
            # Verify input shape
            try:
                expected_shape = model.input_shape
                print(f"Model expects input shape: {expected_shape}")
                if expected_shape != (None, EYE_IMG_SIZE[0], EYE_IMG_SIZE[1], 1):
                     print(f"Warning: Model input shape {expected_shape} differs from expected {(None, EYE_IMG_SIZE[0], EYE_IMG_SIZE[1], 1)}")
            except Exception as e:
                print(f"Could not determine model input shape: {e}")
            return model
        else:
            print(f"No existing model found at {EYE_MODEL_PATH}. Attempting to train/create dummy...")
def main():
    eye_model = load_model()
    if eye_model is None:
        print("Error: Failed to load or create eye state model. Exiting.")
        return

    cap = cv2.VideoCapture(0) # Use default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None or fps < 1:
        print(f"Warning: Invalid FPS detected ({fps}). Setting to fallback value 20.")
        fps = 30 # Fallback FPS
    else:
        print(f"Camera resolution: {frame_width}x{frame_height}, FPS: {fps:.2f}")

    # Calculate frame thresholds based on seconds and FPS
    closed_frames_threshold = int(CLOSED_FRAMES_THRESHOLD_SECONDS * fps)
    print(f"Eye closure frame threshold: {closed_frames_threshold} frames ({CLOSED_FRAMES_THRESHOLD_SECONDS}s)")
    print(f"Yawn duration threshold: {YAWN_DURATION_THRESHOLD_SECONDS}s")

    # State variables
    frame_count = 0
    prediction_history = []
    closed_frames_counter = 0
    yawn_start_time = 0 # Timestamp when yawn started, 0 if not yawning
    eye_closed_start_time = 0 # Timestamp when eyes first closed, 0 if eyes are open
    last_alarm_check_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame. End of stream?")
            break

        current_time = time.time()
        frame_count += 1

        # --- Frame Preprocessing ---
        frame = cv2.flip(frame, 1) # Flip horizontally
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False # Optimize processing

        # --- MediaPipe Face Mesh Detection ---
        results = face_mesh.process(rgb_frame)

        rgb_frame.flags.writeable = True # Re-enable writing
        # Draw on the original BGR frame

        # --- Reset States for Current Frame ---
        face_detected = False
        eyes_detected = False
        eyes_closed_this_frame = False
        yawning_this_frame = False
        ear_value = 0
        lip_distance = 0
        eye_state_pred = 0.5 # Default to uncertain

        # --- Process Face Landmarks ---
        if results.multi_face_landmarks:
            face_detected = True
            # Assuming only one face
            face_landmarks = results.multi_face_landmarks[0]

            # --- Eye State Detection ---
            left_eye_img = get_eye_region(frame, face_landmarks, LEFT_EYE_INDICES)
            right_eye_img = get_eye_region(frame, face_landmarks, RIGHT_EYE_INDICES)

            if left_eye_img is not None and right_eye_img is not None:
                eyes_detected = True
                try:
                    left_ear = calculate_eye_aspect_ratio(face_landmarks, LEFT_EYE_INDICES)
                    right_ear = calculate_eye_aspect_ratio(face_landmarks, RIGHT_EYE_INDICES)
                    ear_value = (left_ear + right_ear) / 2.0

                    left_eye_processed = preprocess_eye_image(left_eye_img)
                    right_eye_processed = preprocess_eye_image(right_eye_img)

                    if left_eye_processed is not None and right_eye_processed is not None:
                        # Predict using the model
                        left_pred = eye_model.predict(np.expand_dims(left_eye_processed, axis=0), verbose=0)[0][0]
                        right_pred = eye_model.predict(np.expand_dims(right_eye_processed, axis=0), verbose=0)[0][0]
                        eye_state_pred = (left_pred + right_pred) / 2.0

                        # Smoothing prediction
                        prediction_history.append(eye_state_pred)
                        if len(prediction_history) > PREDICTION_HISTORY_SIZE:
                            prediction_history.pop(0)
                        smoothed_pred = sum(prediction_history) / len(prediction_history)

                        # Determine eye state (Combine model prediction and EAR)
                        if smoothed_pred < EYE_STATE_THRESHOLD or ear_value < EAR_THRESHOLD:
                            eyes_closed_this_frame = True
                            eye_status_text = f"Closed (P:{smoothed_pred:.2f} E:{ear_value:.1f})"
                            eye_color = (0, 0, 255) # Red
                            if eye_closed_start_time == 0:  # Start timer only if it wasn't already running
                                eye_closed_start_time = current_time
                        else:
                            eyes_closed_this_frame = False
                            eye_status_text = f"Open (P:{smoothed_pred:.2f} E:{ear_value:.1f})"
                            eye_color = (0, 255, 0) # Green
                            if eye_closed_start_time != 0:  # If timer was running, stop and report
                                print(f"Eyes opened after {current_time - eye_closed_start_time:.2f}s")
                                eye_closed_start_time = 0  # Reset eye closed timer

                        cv2.putText(frame, f"Eyes: {eye_status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)

                    else:
                        cv2.putText(frame, "Eyes: Preprocessing Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        eyes_closed_this_frame = True # Treat error as potentially closed

                except Exception as e:
                    print(f"Error processing eyes: {e}")
                    cv2.putText(frame, "Eyes: Processing Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    eyes_closed_this_frame = True # Treat error as potentially closed
            else:
                # Eyes not clearly detected/extracted
                cv2.putText(frame, "Eyes: Not Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                eyes_closed_this_frame = True # Treat undetected eyes as potentially closed

            # --- Yawn Detection ---
            try:
                top_lip = face_landmarks.landmark[MOUTH_INNER_INDICES[0]]
                bottom_lip = face_landmarks.landmark[MOUTH_INNER_INDICES[1]]

                top_lip_pos = (int(top_lip.x * frame_width), int(top_lip.y * frame_height))
                bottom_lip_pos = (int(bottom_lip.x * frame_width), int(bottom_lip.y * frame_height))

                lip_distance = abs(top_lip_pos[1] - bottom_lip_pos[1])

                cv2.putText(frame, f"Mouth Open: {lip_distance:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.line(frame, top_lip_pos, bottom_lip_pos, (0, 255, 255), 1)

                if lip_distance > YAWN_LIP_DISTANCE_THRESHOLD:
                    yawning_this_frame = True
                    if yawn_start_time == 0: # Start timer only if it wasn't already running
                        yawn_start_time = current_time
                        print(f"Yawn Started at {yawn_start_time}")
                else:
                    yawning_this_frame = False
                    if yawn_start_time != 0: # If timer was running, stop and report
                         print(f"Yawn Ended (Duration: {current_time - yawn_start_time:.2f}s")
                         yawn_start_time = 0 # Reset yawn timer
                    # If yawn_start_time was already 0, do nothing

            except (IndexError, AttributeError) as e:
                print(f"Error processing mouth landmarks: {e}")
                cv2.putText(frame, "Mouth: Error", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                yawn_start_time = 0 # Reset on error

            # Draw face mesh landmarks (optional, can be performance intensive)
            # mp.solutions.drawing_utils.draw_landmarks(
            #     image=frame,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_TESSELATION,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())

            # Display eye closed duration
            if eye_closed_start_time != 0:  # If eyes are currently closed
                eye_closed_duration = current_time - eye_closed_start_time
                cv2.putText(frame, f"Closed Time: {eye_closed_duration:.1f}s", 
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Closed Time: 0.0s", 
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
            # No face detected
            cv2.putText(frame, "Face Not Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            face_detected = False
            eyes_closed_this_frame = True # Treat no face as drowsy state (eyes closed)
            yawn_start_time = 0 # Reset yawn timer if face is lost
            eye_closed_start_time = 0  # Also reset eye closed timer
            closed_frames_counter = 0 # Also reset eye counter if face is lost

        # --- Drowsiness State Logic (Refined) ---
        eye_closure_alert = False
        yawn_alert = False

        # Check eye closure duration
        if eyes_closed_this_frame:
            closed_frames_counter += 1
            # print(f"Closed frames: {closed_frames_counter}/{closed_frames_threshold}") # Debug
            if closed_frames_counter >= closed_frames_threshold:
                eye_closure_alert = True
                cv2.putText(frame, "ALERT: EYES CLOSED/UNDETECTED", (10, frame_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            # This is the reset condition for the eye timer
            if closed_frames_counter > 0:
                print(f"Eye closed duration ended (Frames: {closed_frames_counter})")
            closed_frames_counter = 0 # Reset counter if eyes are open or detected

        # Check yawn duration
        yawn_duration = 0
        if yawn_start_time != 0: # Timer is running
            yawn_duration = current_time - yawn_start_time
            cv2.putText(frame, f"Yawn Time: {yawn_duration:.1f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if yawn_alert else (255, 255, 0), 2)
            if yawn_duration >= YAWN_DURATION_THRESHOLD_SECONDS:
                yawn_alert = True
                cv2.putText(frame, "ALERT: PROLONGED YAWN", (10, frame_height - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
             # Timer is not running (already reset or never started)
             cv2.putText(frame, f"Yawn Time: 0.0s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # --- Control Alarm ---
        # Alarm triggers if EITHER condition is met
        trigger_alarm_now = eye_closure_alert or yawn_alert

        if trigger_alarm_now:
            play_alarm()
        else:
            stop_alarm()

        # --- Display Frame ---
        cv2.imshow('Drowsiness Detection', frame)

        # --- Exit Condition ---
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exit key pressed.")
            break

    # --- Cleanup ---
    print("Cleaning up...")
    cap.release()
    stop_alarm() # Ensure alarm is stopped on exit
    if alarm_sound_available:
        pygame.mixer.quit()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("Cleanup complete. Exiting.")

if __name__ == "__main__":
    main()