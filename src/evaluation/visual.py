import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', labels=None):
    """
    Create and return a confusion matrix plot
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        title: Title for the plot
        labels: Class labels (default: [0, 1, 2])
    
    Returns:
        matplotlib figure
    """
    if labels is None:
        labels = [0, 1, 2]
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    
    return fig