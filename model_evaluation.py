"""
Model Evaluation Module for Driver Drowsiness Detection
------------------------------------------------------
This module contains functions for evaluating the performance of both the
eye state detection model and the yawn detection model. It calculates various 
metrics like accuracy, precision, recall, and F1-score, and generates visualization
plots such as confusion matrices, ROC curves, and precision-recall curves.

The evaluation results are saved to a timestamped directory for reference.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_score, recall_score
from sklearn.metrics import f1_score, accuracy_score, roc_curve, roc_auc_score, classification_report
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
from datetime import datetime

# Import model loaders
from eye_state_model import load_dataset as load_eye_dataset
from yawn_detection_model import extract_frames_from_videos

# Constants - file paths for models and labels
# All files are expected to be in the same directory as this script
EYE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "eye_state_model.h5")
EYE_LABEL_PATH = os.path.join(os.path.dirname(__file__), "eye_labels.pkl")
YAWN_MODEL_PATH = os.path.join(os.path.dirname(__file__), "yawn_detection_model.h5")
YAWN_LABEL_PATH = os.path.join(os.path.dirname(__file__), "yawn_labels.pkl")

# Create output directory for results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"evaluation_results_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

def load_model_and_labels(model_path, label_path):
    """
    Load a trained model and its class labels from disk
    
    Parameters:
    -----------
    model_path : str
        Path to the saved Keras model (.h5 file)
    label_path : str
        Path to the saved class labels (.pkl file)
        
    Returns:
    --------
    tuple
        (model, class_labels) or (None, None) if files not found
    """
    if not os.path.exists(model_path) or not os.path.exists(label_path):
        return None, None
    
    model = load_model(model_path)
    with open(label_path, 'rb') as f:
        class_labels = pickle.load(f)
    
    return model, class_labels

def evaluate_eye_model():
    """
    Evaluate the eye state detection model using test data
    
    This function:
    1. Loads the trained eye state model
    2. Loads test data from the eye dataset
    3. Makes predictions on test data
    4. Calculates performance metrics
    5. Generates visualization plots
    """
    print("\n===== Evaluating Eye State Detection Model =====")
    
    # Load model and labels
    model, class_labels = load_model_and_labels(EYE_MODEL_PATH, EYE_LABEL_PATH)
    if model is None:
        print("Eye state model not found. Please train the model first.")
        return
    
    # Load test data
    print("Loading eye state test data...")
    try:
        X, y = load_eye_dataset()
        
        # Split into test set (20%)
        test_split = int(0.8 * len(X))
        X_test, y_test = X[test_split:], y[test_split:]
        
        # Make predictions
        print("Running predictions...")
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        print("Calculating metrics...")
        calculate_and_save_metrics(y_true, y_pred, y_pred_prob, class_labels, "eye_state")
    except Exception as e:
        print(f"Error evaluating eye state model: {e}")

def evaluate_yawn_model():
    """
    Evaluate the yawn detection model using test data
    
    This function:
    1. Loads the trained yawn detection model
    2. Loads test data from the YawDD dataset
    3. Makes predictions on test data
    4. Calculates performance metrics
    5. Generates visualization plots
    """
    print("\n===== Evaluating Yawn Detection Model =====")
    
    # Load model and labels
    model, class_labels = load_model_and_labels(YAWN_MODEL_PATH, YAWN_LABEL_PATH)
    if model is None:
        print("Yawn detection model not found. Please train the model first.")
        return
    
    # Load test data
    print("Loading yawn detection test data...")
    try:
        # YawDD dataset path should be in the same directory
        dataset_path = os.path.join(os.path.dirname(__file__), "YawDD")
        X, y = extract_frames_from_videos(dataset_path, max_frames_per_video=5)
        
        # Split into test set (20%)
        test_split = int(0.8 * len(X))
        X_test, y_test = X[test_split:], y[test_split:]
        
        # Make predictions
        print("Running predictions...")
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        print("Calculating metrics...")
        calculate_and_save_metrics(y_true, y_pred, y_pred_prob, class_labels, "yawn_detection")
    except Exception as e:
        print(f"Error evaluating yawn detection model: {e}")

def calculate_binary_metrics(y_true, y_pred, y_pred_prob, class_labels, model_name):
    """
    Calculate and save metrics for binary classification model
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        Ground truth labels (0 or 1)
    y_pred : numpy.ndarray
        Predicted labels (0 or 1)
    y_pred_prob : numpy.ndarray
        Predicted probabilities for each class
    class_labels : dict
        Dictionary mapping class indices to class names
    model_name : str
        Name of the model being evaluated (for file naming)
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['True Positives', 'True Negatives', 'False Positives', 'False Negatives', 
                  'Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value': [tp, tn, fp, fn, accuracy, precision, recall, f1]
    })
    metrics_df.to_csv(f"{output_dir}/{model_name}_metrics.csv", index=False)
    
    # Print metrics
    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=list(class_labels.values()),
               yticklabels=list(class_labels.values()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name.replace("_", " ").title()}')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_confusion_matrix.png")
    plt.close()
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob[:, 1])
    roc_auc = roc_auc_score(y_true, y_pred_prob[:, 1])
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name.replace("_", " ").title()}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_roc_curve.png")
    plt.close()
    
    # Plot Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_prob[:, 1])
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall Curve - {model_name.replace("_", " ").title()}')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_precision_recall_curve.png")
    plt.close()

def calculate_multiclass_metrics(y_true, y_pred, y_pred_prob, class_labels, model_name):
    """
    Calculate and save metrics for multi-class classification model
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        Ground truth labels
    y_pred : numpy.ndarray
        Predicted labels
    y_pred_prob : numpy.ndarray
        Predicted probabilities for each class
    class_labels : dict
        Dictionary mapping class indices to class names
    model_name : str
        Name of the model being evaluated (for file naming)
    """
    # Calculate metrics for each class
    cm = confusion_matrix(y_true, y_pred)
    class_metrics = {}
    
    for i, class_name in class_labels.items():
        # Create binary classification problem for each class (one-vs-rest)
        y_true_class = (y_true == i).astype(int)
        y_pred_class = (y_pred == i).astype(int)
        y_pred_prob_class = y_pred_prob[:, i]
        
        # Calculate metrics
        tn = np.sum((y_true_class == 0) & (y_pred_class == 0))
        fp = np.sum((y_true_class == 0) & (y_pred_class == 1))
        fn = np.sum((y_true_class == 1) & (y_pred_class == 0))
        tp = np.sum((y_true_class == 1) & (y_pred_class == 1))
        
        accuracy = accuracy_score(y_true_class, y_pred_class)
        precision = precision_score(y_true_class, y_pred_class, zero_division=0)
        recall = recall_score(y_true_class, y_pred_class, zero_division=0)
        f1 = f1_score(y_true_class, y_pred_class, zero_division=0)
        
        class_metrics[class_name] = {
            'True Positives': tp,
            'True Negatives': tn,
            'False Positives': fp,
            'False Negatives': fn,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }
        
        # Plot ROC and precision-recall curves for each class
        plot_class_curves(y_true_class, y_pred_prob_class, model_name, class_name)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(class_metrics).T
    metrics_df.to_csv(f"{output_dir}/{model_name}_metrics_by_class.csv")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=list(class_labels.values())))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=list(class_labels.values()),
               yticklabels=list(class_labels.values()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name.replace("_", " ").title()}')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_confusion_matrix.png")
    plt.close()
    
    # Plot overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    plt.bar(['Accuracy'], [accuracy], color='blue')
    plt.ylim([0, 1.0])
    plt.title(f'Overall Accuracy - {model_name.replace("_", " ").title()}')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_accuracy.png")
    plt.close()

def plot_class_curves(y_true_class, y_pred_prob_class, model_name, class_name):
    """
    Plot ROC and Precision-Recall curves for a specific class
    
    Parameters:
    -----------
    y_true_class : numpy.ndarray
        Binary ground truth labels for the class (0 or 1)
    y_pred_prob_class : numpy.ndarray
        Predicted probabilities for the class
    model_name : str
        Name of the model being evaluated
    class_name : str
        Name of the class being evaluated
    """
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true_class, y_pred_prob_class)
    roc_auc = roc_auc_score(y_true_class, y_pred_prob_class)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name.replace("_", " ").title()} - {class_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_{class_name}_roc_curve.png")
    plt.close()
    
    # Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_true_class, y_pred_prob_class)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall Curve - {model_name.replace("_", " ").title()} - {class_name}')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_{class_name}_precision_recall_curve.png")
    plt.close()

def calculate_and_save_metrics(y_true, y_pred, y_pred_prob, class_labels, model_name):
    """
    Calculate metrics and save visualizations based on classification type
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        Ground truth labels
    y_pred : numpy.ndarray
        Predicted labels
    y_pred_prob : numpy.ndarray
        Predicted probabilities for each class
    class_labels : dict
        Dictionary mapping class indices to class names
    model_name : str
        Name of the model being evaluated
    """
    # Determine if binary or multi-class classification
    if len(class_labels) == 2:
        calculate_binary_metrics(y_true, y_pred, y_pred_prob, class_labels, model_name)
    else:
        calculate_multiclass_metrics(y_true, y_pred, y_pred_prob, class_labels, model_name)

def run_evaluation():
    """
    Run the evaluation for both the eye state and yawn detection models
    and save results to the output directory
    """
    print(f"Evaluation results will be saved to: {output_dir}")
    evaluate_eye_model()
    evaluate_yawn_model()
    print(f"\nEvaluation complete. Results saved to: {output_dir}")

if __name__ == "__main__":
    run_evaluation() 