# yawn_detection_training.py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
import os
import pickle
import cv2
import numpy as np

# Configuration
IMG_SIZE = (96, 96)
INPUT_SHAPE = (*IMG_SIZE, 3)
NUM_CLASSES = 2  # yawn vs not_yawn
BATCH_SIZE = 32
EPOCHS = 20
MODEL_SAVE_PATH = "yawn_state_model.h5"
LABELS_SAVE_PATH = "yawn_labels.pkl"

# Dataset Configuration
# Expected directory structure:
# combined_yawn_dataset/
#   train/
#     yawn/       (frames extracted from YawwDD yawning videos and UL-RDD Drowsiness videos)
#     not_yawn/   (frames from YawwDD normal/talking and UL-RDD Not_Drowsiness videos)
#   validation/
#     yawn/
#     not_yawn/
BASE_DATA_DIR = "Yaww_DD"  # <-- User must set this path to organized folder
TRAIN_DIR = os.path.join(BASE_DATA_DIR, "train")
VAL_DIR = os.path.join(BASE_DATA_DIR, "validation")

def extract_frames_from_videos(video_dir, output_dir, class_name, max_frames_per_video=50):
    """Helper function to extract frames from videos and organize them"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for video_file in os.listdir(video_dir):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(video_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            
            while frame_count < max_frames_per_video:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize and save frame
                frame = cv2.resize(frame, IMG_SIZE)
                frame_path = os.path.join(output_dir, f"{video_file}_frame{frame_count}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_count += 1
            
            cap.release()

def prepare_datasets():
    """User needs to run this once to prepare the dataset from original videos"""
    # This is a one-time setup function
    # Structure for YawwDD:
    # For yawn class: Extract frames from yawning segments in Male/Female folders
    # For not_yawn: Extract frames from normal/talking segments
    
    # Structure for UL-RDD:
    # Drowsiness videos -> yawn class
    # Not_Drowsiness videos -> not_yawn class
    
    print("This function needs to be run once to prepare the dataset from original videos")
    print("User should manually extract frames from:")
    print("- YawwDD: yawning segments -> yawn class")
    print("- YawwDD: normal/talking segments -> not_yawn class")
    print("- UL-RDD Drowsiness videos -> yawn class")
    print("- UL-RDD Not_Drowsiness videos -> not_yawn class")

def create_model():
    base_model = MobileNetV2(input_shape=INPUT_SHAPE, include_top=False, weights='imagenet')
    base_model.trainable = False
    
    model = models.Sequential([
        layers.Input(shape=INPUT_SHAPE),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def setup_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    print("Loading combined yawn dataset (YawwDD and UL-RDD)...")
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=['not_yawn', 'yawn']
    )
    
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=['not_yawn', 'yawn'],
        shuffle=False
    )
    
    # Save label mapping
    with open(LABELS_SAVE_PATH, 'wb') as f:
        pickle.dump(train_gen.class_indices, f)
    
    return train_gen, val_gen

def train():
    print("Training yawn detection model on YawwDD and UL-RDD datasets...")
    train_gen, val_gen = setup_data_generators()
    model = create_model()
    
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        validation_data=val_gen,
        validation_steps=val_gen.samples // BATCH_SIZE,
        epochs=EPOCHS
    )
    
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    return history

if __name__ == "__main__":
    # Uncomment to prepare datasets (one-time)
    # prepare_datasets()
    train()