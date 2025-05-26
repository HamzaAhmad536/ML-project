import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pickle
import cv2

# Configuration
IMG_SIZE = (96, 96)
INPUT_SHAPE = (*IMG_SIZE, 3)
NUM_CLASSES = 2  # open_eye, closed_eye
BATCH_SIZE = 32
EPOCHS = 15
MODEL_SAVE_PATH = "eye_state_model.h5"
LABELS_SAVE_PATH = "eye_labels.pkl"

# MRL_EYE Dataset Configuration
# Expected directory structure:
# mrl_eye_dataset/
#   train/
#     open_eye/  (contains awake PNGs from MRL_EYE)
#     closed_eye/ (contains sleepy PNGs from MRL_EYE)
#   validation/
#     open_eye/
#     closed_eye/
BASE_DATA_DIR = "MRL_EYE"  # <-- User must set this path to MRL_EYE organized folder
TRAIN_DIR = os.path.join(BASE_DATA_DIR, "train")
VAL_DIR = os.path.join(BASE_DATA_DIR, "validation")

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
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    print("Loading MRL_EYE dataset images...")
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=['open_eye', 'closed_eye']  # Maps MRL_EYE awake->open_eye, sleepy->closed_eye
    )
    
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=['open_eye', 'closed_eye'],
        shuffle=False
    )
    
    # Save label mapping
    with open(LABELS_SAVE_PATH, 'wb') as f:
        pickle.dump(train_gen.class_indices, f)
    
    return train_gen, val_gen

def train():
    print("Training eye state model on MRL_EYE dataset...")
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
    train()