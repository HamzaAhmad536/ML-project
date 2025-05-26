"""
Preprocessing Module for Driver Drowsiness Detection
---------------------------------------------------
This module contains functions for preprocessing images to enhance features
for drowsiness detection, including contrast enhancement, noise reduction,
and preparation of eye/mouth region images for model input.
"""

import cv2
import numpy as np

def preprocess_image(image):
    """
    Preprocess the main camera frame to enhance features for drowsiness detection.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image in BGR format from the camera
        
    Returns:
    --------
    numpy.ndarray
        Enhanced image with improved contrast and reduced noise
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization for better contrast
    # CLAHE (Contrast Limited Adaptive Histogram Equalization) gives better results 
    # than standard histogram equalization for facial features
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(gray)
    
    # Apply bilateral filter to reduce noise while preserving edges
    # Parameters: image, diameter of pixel neighborhood, sigma color, sigma space
    denoised = cv2.bilateralFilter(equalized, 9, 75, 75)
    
    # Convert back to BGR for MediaPipe (MediaPipe expects BGR input)
    enhanced_image = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    
    return enhanced_image

def preprocess_eye_for_detection(eye_image):
    """
    Preprocess eye image specifically for the eye state detection model.
    Enhances contrast and prepares the image for input to MobileNetV2.
    
    Parameters:
    -----------
    eye_image : numpy.ndarray
        Cropped eye region image
        
    Returns:
    --------
    numpy.ndarray or None
        Preprocessed eye image ready for model input, or None if input is invalid
    """
    # Check if the image is valid
    if eye_image is None or eye_image.size == 0:
        return None
    
    # Convert to RGB (MobileNetV2 expects RGB input)
    if len(eye_image.shape) == 2:  # If grayscale
        eye_rgb = cv2.cvtColor(eye_image, cv2.COLOR_GRAY2RGB)
    else:  # If BGR
        eye_rgb = cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB)
    
    # Apply contrast enhancement in LAB color space
    # L channel represents lightness, A channel represents green-red, B channel represents blue-yellow
    lab = cv2.cvtColor(eye_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to lightness channel only
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge channels back
    enhanced_lab = cv2.merge((cl, a, b))
    
    # Convert back to RGB for model input
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb 