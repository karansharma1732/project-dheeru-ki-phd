import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def plot_training_curves(history, save_path=None):
    """Plot training and validation curves"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy curves
    axes[1].plot(history['train_accuracy'], label='Train Accuracy', marker='o')
    axes[1].plot(history['val_accuracy'], label='Val Accuracy', marker='s')
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(predictions, labels, class_names=None, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_data_distribution(partition_stats, save_path=None):
    """Plot data distribution across clients"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Total samples per client
    clients = list(partition_stats.keys())
    total_samples = [stats['total_samples'] for stats in partition_stats.values()]
    
    axes[0].bar(range(len(clients)), total_samples)
    axes[0].set_xlabel('Client')
    axes[0].set_ylabel('Number of Samples')
    axes[0].set_title('Data Distribution Across Clients')
    axes[0].set_xticks(range(len(clients)))
    axes[0].set_xticklabels([f"C{i}" for i in range(len(clients))], rotation=45)
    
    # Class distribution per client
    class_distributions = []
    for stats in partition_stats.values():
        class_dist = stats['class_distribution']
        class_distributions.append([class_dist.get(i, 0) for i in range(max(class_dist.keys()) + 1)])
    
    class_distributions = np.array(class_distributions).T
    
    x = np.arange(len(clients))
    width = 0.35
    
    for i, class_dist in enumerate(class_distributions):
        axes[1].bar(x + i * width, class_dist, width, label=f'Class {i}')
    
    axes[1].set_xlabel('Client')
    axes[1].set_ylabel('Number of Samples')
    axes[1].set_title('Class Distribution Across Clients')
    axes[1].set_xticks(x + width / 2)
    axes[1].set_xticklabels([f"C{i}" for i in range(len(clients))], rotation=45)
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()