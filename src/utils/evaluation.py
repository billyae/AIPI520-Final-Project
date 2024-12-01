from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, dataloader, device, threshold=0.5):
    """Evaluate model performance for multi-label classification.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader containing test data
        device: Device to run evaluation on
        threshold: Threshold for converting probabilities to binary predictions
        
    Returns:
        dict: Dictionary containing various metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in dataloader:
            images = data['image'].to(device)
            labels = data['labels']
            
            outputs = model(images)
            predictions = (outputs > threshold).float().cpu()
            
            all_preds.append(predictions)
            all_labels.append(labels)
    
    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )
    
    # Calculate average precision
    ap = average_precision_score(all_labels, all_preds, average='macro')
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'average_precision': ap,
        'predictions': all_preds,
        'true_labels': all_labels
    }

def plot_metrics(metrics, label_names, save_path):
    """Plot and save evaluation metrics visualization.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        label_names: List of label names
        save_path: Path to save the plots
    """
    # Create precision-recall plot for each class
    plt.figure(figsize=(12, 8))
    
    for i, label in enumerate(label_names):
        precision = metrics['precision'][i]
        recall = metrics['recall'][i]
        plt.scatter(recall, precision, label=label)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall by Class')
    plt.legend()
    plt.savefig(f'{save_path}_pr_curve.png')
    plt.close()
