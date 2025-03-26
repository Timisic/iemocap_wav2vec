import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from config import Config

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, mode='max'):
        """
        Early stopping to prevent overfitting
        
        Args:
            patience: Number of epochs to wait before early stopping
            min_delta: Minimum change in monitored value to qualify as an improvement
            mode: 'min' for loss, 'max' for metrics like accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        
    def __call__(self, value):
        if self.best_value is None:
            self.best_value = value
            return False
            
        if self.mode == 'min':
            if value < self.best_value - self.min_delta:
                self.best_value = value
                self.counter = 0
            else:
                self.counter += 1
        else:
            if value > self.best_value + self.min_delta:
                self.best_value = value
                self.counter = 0
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            self.early_stop = True
            return True
            
        return False


class MetricsTracker:
    def __init__(self, config, figure_dir=None):
        """
        Track and plot training metrics
        
        Args:
            config: Configuration object
            figure_dir: 图片保存目录
        """
        self.config = config
        self.figure_dir = figure_dir or config.FIGURE_DIR
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
    def update_train(self, loss, metrics):
        """Update training metrics"""
        self.train_losses.append(loss)
        self.train_metrics.append(metrics)
        
    def update_val(self, loss, metrics):
        """Update validation metrics"""
        self.val_losses.append(loss)
        self.val_metrics.append(metrics)
        
    def plot_losses(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.figure_dir / 'loss_plot.png')
        plt.close()
        
    def plot_metrics(self, metric_name):
        """
        Plot specific metric
        
        Args:
            metric_name: Name of the metric to plot
        """
        train_values = [m[metric_name] for m in self.train_metrics]
        val_values = [m[metric_name] for m in self.val_metrics]
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_values, label=f'Training {metric_name}')
        plt.plot(val_values, label=f'Validation {metric_name}')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(f'Training and Validation {metric_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.figure_dir / f'{metric_name}_plot.png')
        plt.close()
        
    def plot_confusion_matrix(self, true_labels, pred_labels):
        """
        Plot confusion matrix
        
        Args:
            true_labels: True labels
            pred_labels: Predicted labels
        """
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.savefig(self.figure_dir / 'confusion_matrix.png')
        plt.close()
        
    def plot_all(self):
        """Plot all metrics"""
        self.plot_losses()
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            self.plot_metrics(metric)


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(predictions, labels):
    """
    Compute evaluation metrics
    
    Args:
        predictions: Model predictions (N, num_classes)
        labels: True labels (N,)
        
    Returns:
        Dictionary containing metrics
    """
    # Get predicted classes
    pred_ids = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, pred_ids)
    precision = precision_score(labels, pred_ids, average='macro', zero_division=1)
    recall = recall_score(labels, pred_ids, average='macro', zero_division=1)
    f1 = f1_score(labels, pred_ids, average='macro', zero_division=1)
    
    # 添加weighted F1计算
    weighted_f1 = f1_score(labels, pred_ids, average='weighted')
    
    # 添加f1_per_class计算
    f1_per_class = f1_score(labels, pred_ids, average=None)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'weighted_f1': weighted_f1,
        'f1_angry': f1_per_class[0],
        'f1_happy': f1_per_class[1],
        'f1_sad': f1_per_class[2],
        'f1_neutral': f1_per_class[3]
    }
    return metrics


if __name__ == "__main__":
    # 测试早停策略
    early_stopping = EarlyStopping(patience=3, mode='max')
    scores = [0.5, 0.6, 0.7, 0.65, 0.64, 0.63]
    
    print("测试早停策略:")
    for i, score in enumerate(scores):
        should_stop = early_stopping(score)
        print(f"轮次 {i+1}, 分数: {score}, 是否应该停止: {should_stop}")
    
    # 测试指标跟踪器
    config = Config()
    tracker = MetricsTracker(config)
    
    # 模拟训练过程
    for i in range(5):
        train_loss = 1.0 - 0.1 * i
        val_loss = 0.9 - 0.08 * i
        
        train_metrics = {
            'accuracy': 0.7 + 0.05 * i,
            'f1': 0.65 + 0.06 * i
        }
        
        val_metrics = {
            'accuracy': 0.65 + 0.06 * i,
            'f1': 0.6 + 0.07 * i
        }
        
        tracker.update_train(train_loss, train_metrics)
        tracker.update_val(val_loss, val_metrics)
    
    # 测试绘图功能
    print("测试绘图功能:")
    tracker.plot_losses()
    print(f"损失曲线已保存到: {config.FIGURE_DIR / 'loss_plot.png'}")
    
    tracker.plot_metrics('accuracy')
    print(f"准确率曲线已保存到: {config.FIGURE_DIR / 'accuracy_plot.png'}")
    
    # 测试混淆矩阵绘制
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 3, 3, 0, 2, 2, 3])
    
    tracker.plot_confusion_matrix(y_true, y_pred)
    print(f"混淆矩阵已保存到: {config.FIGURE_DIR / 'confusion_matrix.png'}")
    
    # 测试指标计算
    preds = np.array([
        [0.8, 0.1, 0.05, 0.05],
        [0.1, 0.7, 0.1, 0.1],
        [0.1, 0.1, 0.6, 0.2],
        [0.05, 0.05, 0.1, 0.8],
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.6, 0.2, 0.1],
        [0.1, 0.1, 0.7, 0.1],
        [0.1, 0.1, 0.1, 0.7]
    ])
    
    labels = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    
    metrics = compute_metrics(preds, labels)
    print("计算的指标:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}") 