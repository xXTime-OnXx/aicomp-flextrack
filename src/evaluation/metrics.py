"""
Evaluation metrics for FlexTrack Challenge
"""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from typing import Dict, Tuple


def geometric_mean_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Geometric Mean Score (primary metric for warm-up phase)
    
    GMS = sqrt(TPR * TNR)
    where:
    - TPR (True Positive Rate) = TP / (TP + FN)
    - TNR (True Negative Rate) = TN / (TN + FP)
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Geometric mean score
    """
    # For multiclass, we need to handle this differently
    # Convert to binary: DR event (1 or -1) vs no event (0)
    y_true_binary = (y_true != 0).astype(int)
    y_pred_binary = (y_pred != 0).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle edge cases
        if cm.shape == (1, 1):
            if y_true_binary[0] == 0:
                tn, fp, fn, tp = cm[0, 0], 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
        else:
            return 0.0
    
    # Calculate TPR and TNR
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Geometric mean
    gms = np.sqrt(tpr * tnr)
    
    return gms


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Primary metric: Geometric Mean Score
    metrics['geometric_mean_score'] = geometric_mean_score(y_true, y_pred)
    
    # Secondary metric: F1 Score
    metrics['f1_score_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_score_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Accuracy
    metrics['accuracy'] = (y_true == y_pred).mean()
    
    # Per-class metrics
    for label in [-1, 0, 1]:
        label_mask = y_true == label
        if label_mask.sum() > 0:
            label_accuracy = (y_pred[label_mask] == label).mean()
            metrics[f'accuracy_class_{label}'] = label_accuracy
    
    # Calculate TPR and TNR for binary classification
    y_true_binary = (y_true != 0).astype(int)
    y_pred_binary = (y_pred != 0).astype(int)
    
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['tnr'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return metrics


def print_evaluation_report(y_true: np.ndarray, y_pred: np.ndarray, verbose: bool = True):
    """
    Print comprehensive evaluation report
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        verbose: Whether to print detailed classification report
    """
    metrics = calculate_metrics(y_true, y_pred)
    
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    
    print(f"\n🎯 PRIMARY METRIC:")
    print(f"  Geometric Mean Score: {metrics['geometric_mean_score']:.4f}")
    
    print(f"\n📊 SECONDARY METRICS:")
    print(f"  F1 Score (Macro):     {metrics['f1_score_macro']:.4f}")
    print(f"  F1 Score (Weighted):  {metrics['f1_score_weighted']:.4f}")
    print(f"  Accuracy:             {metrics['accuracy']:.4f}")
    
    print(f"\n📈 BINARY CLASSIFICATION METRICS:")
    if 'tpr' in metrics:
        print(f"  TPR (Sensitivity):    {metrics['tpr']:.4f}")
        print(f"  TNR (Specificity):    {metrics['tnr']:.4f}")
        print(f"  FPR:                  {metrics['fpr']:.4f}")
        print(f"  FNR:                  {metrics['fnr']:.4f}")
    
    print(f"\n📋 PER-CLASS ACCURACY:")
    for label in [-1, 0, 1]:
        if f'accuracy_class_{label}' in metrics:
            label_name = {-1: "Decrease", 0: "Baseline", 1: "Increase"}[label]
            print(f"  Class {label:2d} ({label_name:8s}): {metrics[f'accuracy_class_{label}']:.4f}")
    
    if verbose:
        print(f"\n📝 DETAILED CLASSIFICATION REPORT:")
        print(classification_report(y_true, y_pred, target_names=['Decrease (-1)', 'Baseline (0)', 'Increase (1)']))
    
    print("="*60 + "\n")
    
    return metrics