"""
Advanced early stopping with multi-metric support.
Prevents overfitting by tracking multiple metrics simultaneously.
"""

import numpy as np
from pathlib import Path
import torch
import json


class MultiMetricEarlyStopping:
    """
    Early stopping that considers multiple metrics with weighted scores.
    
    Advantages over single-metric stopping:
    - Prevents overfitting on one metric while degrading others
    - Better general model performance
    - More stable convergence
    
    Example metric weights:
        {'loss': 0.2, 'accuracy': 0.4, 'bertscore': 0.3, 'f1': 0.1}
    """
    
    def __init__(self, patience=5, metric_weights=None, mode='maximize',
                 save_dir=None, verbose=True):
        """
        Args:
            patience: Number of evaluations with no improvement before stopping
            metric_weights: Dict of {metric_name: weight}. If None, uses 'loss' only
            mode: 'maximize' or 'minimize'
            save_dir: Directory to save best model
            verbose: Print progress
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_metrics = None
        self.save_dir = Path(save_dir) if save_dir else None
        self.verbose = verbose
        self.mode = mode
        
        # Default metric weights if not provided
        if metric_weights is None:
            self.metric_weights = {'loss': 1.0}
        else:
            self.metric_weights = metric_weights
            # Normalize weights to sum to 1
            total_weight = sum(self.metric_weights.values())
            self.metric_weights = {k: v/total_weight for k, v in self.metric_weights.items()}
        
        self.history = []
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_score(self, metrics):
        """
        Compute weighted score from multiple metrics.
        
        Args:
            metrics: Dict of metric_name -> value
        
        Returns:
            Weighted score
        """
        score = 0.0
        
        for metric_name, weight in self.metric_weights.items():
            if metric_name not in metrics:
                if self.verbose:
                    print(f"[WARNING] Metric '{metric_name}' not found in current metrics")
                continue
            
            metric_value = metrics[metric_name]
            
            # Handle loss (we want to minimize it)
            if 'loss' in metric_name.lower():
                # Invert loss for maximization context
                metric_contribution = -metric_value if self.mode == 'maximize' else metric_value
            else:
                # Most metrics should be maximized (accuracy, F1, etc.)
                metric_contribution = metric_value
            
            score += metric_contribution * weight
        
        return score
    
    def __call__(self, metrics, model=None, epoch=None):
        """
        Check if should stop training.
        
        Args:
            metrics: Dict of metric_name -> value
            model: Model to save if best
            epoch: Current epoch number
        
        Returns:
            True if should stop, False otherwise
        """
        score = self.compute_score(metrics)
        
        # Store history
        self.history.append({
            'epoch': epoch,
            'score': score,
            'metrics': metrics.copy()
        })
        
        if self.best_score is None:
            self.best_score = score
            self.best_metrics = metrics.copy()
            if model is not None and self.save_dir:
                self._save_checkpoint(model, epoch, metrics)
        elif score > self.best_score:
            self.best_score = score
            self.best_metrics = metrics.copy()
            self.counter = 0
            if model is not None and self.save_dir:
                self._save_checkpoint(model, epoch, metrics)
            if self.verbose:
                print(f"✓ Epoch {epoch}: New best score {score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"✗ Epoch {epoch}: No improvement ({self.counter}/{self.patience})")
        
        # Check if should stop
        if self.counter >= self.patience:
            if self.verbose:
                print(f"\n[EARLY STOPPING] Patience exceeded. Best metrics:")
                for k, v in self.best_metrics.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
            return True
        
        return False
    
    def _save_checkpoint(self, model, epoch, metrics):
        """Save best model checkpoint."""
        if self.save_dir is None:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics
        }
        
        save_path = self.save_dir / f"best_checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, save_path)
        
        # Also save metrics record
        metrics_path = self.save_dir / f"best_metrics_epoch_{epoch}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        if self.verbose:
            print(f"  💾 Saved checkpoint to {save_path}")
    
    def get_best_metrics(self):
        """Return best metrics found during training."""
        return self.best_metrics
    
    def get_history(self):
        """Return training history."""
        return self.history
    
    def plot_metrics(self, save_path=None):
        """
        Plot metric progression during training.
        
        Args:
            save_path: Path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[WARNING] matplotlib not installed, cannot plot")
            return
        
        if not self.history:
            print("[WARNING] No history to plot")
            return
        
        epochs = [h['epoch'] for h in self.history]
        scores = [h['score'] for h in self.history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, scores, 'b-o', label='Composite Score')
        plt.axhline(y=self.best_score, color='r', linestyle='--', label=f'Best: {self.best_score:.4f}')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.title('Early Stopping - Composite Metric Score')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[INFO] Metric plot saved to {save_path}")
        
        plt.close()


class DynamicClassWeights:
    """
    Compute class weights dynamically from training data.
    Adapts to actual data distribution.
    """
    
    @staticmethod
    def compute_weights(dataloader, device='cpu'):
        """
        Compute class weights from data distribution.
        
        Args:
            dataloader: DataLoader to analyze
            device: Device for tensor
        
        Returns:
            Tensor of class weights
        """
        class_counts = {}
        
        for batch in dataloader:
            labels = batch.get('label_closed', None)
            if labels is None:
                continue
            
            # Count occurrences of each class
            unique_labels, counts = torch.unique(labels, return_counts=True)
            for label, count in zip(unique_labels, counts):
                label_idx = label.item()
                if label_idx >= 0:  # Ignore negative indices
                    class_counts[label_idx] = class_counts.get(label_idx, 0) + count.item()
        
        if not class_counts:
            # Default weights if no data found
            return torch.ones(2, device=device)
        
        # Compute inverse frequency weights
        total_samples = sum(class_counts.values())
        num_classes = len(class_counts)
        
        weights = torch.zeros(max(class_counts.keys()) + 1, device=device)
        for class_idx, count in class_counts.items():
            # Weight = total / (num_classes * count) - higher weight for rarer classes
            weight = total_samples / (num_classes * max(count, 1))
            weights[class_idx] = weight
        
        # Normalize to sum to num_classes
        weights = weights / weights.sum() * num_classes
        
        print("[INFO] Dynamic Class Weights:")
        for class_idx in sorted(class_counts.keys()):
            print(f"  Class {class_idx}: Weight={weights[class_idx]:.4f}, Samples={class_counts[class_idx]}")
        
        return weights.to(device)
