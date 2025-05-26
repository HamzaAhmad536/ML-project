# Training Script for Drowsiness Detection Model (3-Class: Open Eye, Closed Eye, Yawn)

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os
import pickle

# --- Import User-Provided Modules ---
# Ensure preprocessing.py and model_evaluation.py are in the same directory or accessible in PYTHONPATH
try:
    import preprocessing
    print("Successfully imported preprocessing module.")
except ImportError:
    print("Error: Could not import preprocessing.py. Make sure it is in the same directory.")
    preprocessing = None # Set to None to handle potential errors later

try:
    import model_evaluation
    print("Successfully imported model_evaluation module.")
except ImportError:
    print("Error: Could not import model_evaluation.py. Make sure it is in the same directory.")
    model_evaluation = None # Set to None

# --- Configuration ---
IMG_WIDTH, IMG_HEIGHT = 96, 96 # MobileNetV2 preferred input size
IMG_CHANNELS = 3 # MobileNetV2 expects 3 channels (RGB)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
NUM_CLASSES = 3
CLASS_NAMES = ["closed_eye", "open_eye", "yawn"] # Alphabetical order often used by flow_from_directory
MODEL_SAVE_PATH = "drowsiness_mobilenetv2_model.h5"
LABELS_SAVE_PATH = "drowsiness_labels.pkl" # To save class indices mapping

# --- Dataset Configuration (User needs to set this path) ---
# User must organize MRL_EYE (awake/sleepy) and pre-extracted YawwDD yawn frames here:
# main_dataset_folder/
#   train/
#     open_eye/  (contains awake PNGs from MRL_EYE)
#     closed_eye/ (contains sleepy PNGs from MRL_EYE)
#     yawn/       (contains pre-extracted yawn image frames from YawwDD)
#   validation/
#     open_eye/   (...)
#     closed_eye/  (...)
#     yawn/        (...)
BASE_DATA_DIR = "main_dataset_folder" # <--- USER MUST SET THIS PATH
TRAIN_DATA_DIR = os.path.join(BASE_DATA_DIR, "train")
VALIDATION_DATA_DIR = os.path.join(BASE_DATA_DIR, "validation")

# --- Training Parameters ---
BATCH_SIZE = 32
EPOCHS = 20 # Adjust as needed
LEARNING_RATE = 0.001

# --- Preprocessing Function Wrapper ---
def apply_preprocessing(image):
    """Applies the preprocessing function from the user module."""
    if preprocessing and hasattr(preprocessing, "preprocess_eye_for_detection"):
        # Ensure image is in the format expected by preprocess_eye_for_detection (e.g., BGR or RGB)
        # ImageDataGenerator loads as RGB by default
        # The user function preprocess_eye_for_detection expects BGR/RGB and returns RGB
        return preprocessing.preprocess_eye_for_detection(image)
    else:
        # Fallback if module or function not found - just return the image
        # Normalization (rescale=1./255) will still be applied by ImageDataGenerator
        print("Warning: Preprocessing function not found. Skipping custom preprocessing.")
        return image

# --- Model Definition ---
def create_mobilenetv2_model(input_shape, num_classes):
    """Creates a 3-class classification model using MobileNetV2 as a base."""
    base_model = MobileNetV2(input_shape=input_shape,
                             include_top=False,
                             weights='imagenet')

    # Freeze the base model layers initially
    base_model.trainable = False

    # Add custom classification layers
    inputs = tf.keras.Input(shape=input_shape)
    # Ensure input is scaled according to MobileNetV2 expectations if not done by generator
    # x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs) # Alternative to rescale=1./255
    x = inputs # Assuming rescale=1./255 is used in generator
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x) # Softmax for multi-class

    model = models.Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy', # Categorical for multi-class
                  metrics=['accuracy'])
    return model

# --- Data Loading and Augmentation ---
def setup_data_generators(train_dir, val_dir, img_height, img_width, batch_size):
    """Sets up data generators for training and validation."""
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        print(f"Error: Training ({train_dir}) or Validation ({val_dir}) directory not found.")
        print("Please ensure the BASE_DATA_DIR is set correctly and contains train/validation subdirectories with class folders.")
        return None, None

    # Apply user\s preprocessing function if available
    preprocessing_func = apply_preprocessing if preprocessing else None

    train_datagen = ImageDataGenerator(
        rescale=1./255, # Normalize pixel values to [0, 1]
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        preprocessing_function=preprocessing_func
    )

    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=preprocessing_func
    )

    print(f"Looking for training data in: {train_dir}")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical', # Use categorical for multi-class
        color_mode='rgb', # Ensure RGB for MobileNetV2 and preprocessing function
        classes=CLASS_NAMES # Optional: Enforce class order if needed
    )

    print(f"Looking for validation data in: {val_dir}")
    validation_generator = validation_datagen.flow_from_directory(
        val_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb',
        classes=CLASS_NAMES, # Optional: Enforce class order
        shuffle=False
    )

    # Save the class indices mapping
    class_indices = train_generator.class_indices
    print(f"Class Indices Found: {class_indices}")
    # Ensure it matches CLASS_NAMES order if specified, otherwise use found order
    labels = dict((v,k) for k,v in class_indices.items())
    with open(LABELS_SAVE_PATH, 'wb') as f:
        pickle.dump(labels, f)
    print(f"Saved class labels mapping to {LABELS_SAVE_PATH}")

    return train_generator, validation_generator

# --- Training Function ---
def train_model(model, train_generator, validation_generator, epochs, model_save_path):
    """Trains the model."""
    print("Starting model training...")

    # Calculate steps per epoch
    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = validation_generator.samples // validation_generator.batch_size
    if steps_per_epoch == 0 or validation_steps == 0:
        print("Warning: Not enough samples found for batch size. Adjust BATCH_SIZE or add more data.")
        # Optionally adjust steps to 1 if samples < batch_size to run anyway
        steps_per_epoch = max(1, steps_per_epoch)
        validation_steps = max(1, validation_steps)

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps
    )

    print(f"Training complete. Saving model to {model_save_path}...")
    model.save(model_save_path)
    print("Model saved successfully.")
    return history

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Drowsiness Detection Model Training (3-Class) --- ")

    # 1. Setup Data Generators
    print("Setting up data generators...")
    train_gen, val_gen = setup_data_generators(
        TRAIN_DATA_DIR, VALIDATION_DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE
    )

    if train_gen is None or val_gen is None:
        print("Failed to set up data generators. Exiting.")
        exit()

    print(f"Found {train_gen.samples} training images belonging to {len(train_gen.class_indices)} classes.")
    print(f"Found {val_gen.samples} validation images belonging to {len(val_gen.class_indices)} classes.")

    if len(train_gen.class_indices) != NUM_CLASSES:
        print(f"Warning: Expected {NUM_CLASSES} classes, but found {len(train_gen.class_indices)} in training data.")
        print("Please check your dataset structure in:", TRAIN_DATA_DIR)
        # Decide whether to exit or proceed
        # exit()

    # 2. Create Model
    print("Creating MobileNetV2 based model...")
    model = create_mobilenetv2_model(INPUT_SHAPE, NUM_CLASSES)
    model.summary()

    # 3. Train Model
    history = train_model(model, train_gen, val_gen, EPOCHS, MODEL_SAVE_PATH)

    # 4. Evaluate Model (using the imported module)
    if model_evaluation and hasattr(model_evaluation, "run_evaluation"):
        print("--- Running Model Evaluation --- ")
        print("Note: The provided model_evaluation.py might need adjustments")
        print("to work correctly with the new 3-class model and combined dataset structure.")
        try:
            # This call assumes model_evaluation.py can find the newly saved model
            # and potentially the data/labels, or has its own evaluation data loading.
            # It might fail if it expects separate models or different data paths.
            model_evaluation.run_evaluation() 
        except Exception as e:
            print(f"Error running model_evaluation: {e}")
            print("Please check model_evaluation.py for compatibility with the 3-class model.")
    else:
        print("Skipping model evaluation as the module was not imported or run_evaluation function is missing.")

    print("--- Training Script Finished --- ")

