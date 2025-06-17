#!/usr/bin/env python3
"""
Training script for Snap Circuit component detection using DSC loss.
DSC loss helps improve bounding box accuracy by focusing on IoU overlap.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.loss import BboxLoss
import numpy as np


class DSCLoss(nn.Module):
    """
    Dice Similarity Coefficient Loss for bounding box regression.
    Focuses on IoU overlap to improve bounding box accuracy.
    """
    
    def __init__(self, smooth=1e-6):
        super(DSCLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred_boxes, target_boxes):
        """
        Calculate DSC loss between predicted and target bounding boxes.
        
        Args:
            pred_boxes: Predicted bounding boxes [N, 4] (x1, y1, x2, y2)
            target_boxes: Target bounding boxes [N, 4] (x1, y1, x2, y2)
        
        Returns:
            DSC loss value
        """
        # Calculate intersection
        x1_inter = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        y1_inter = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        x2_inter = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        y2_inter = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        
        # Calculate intersection area
        inter_area = torch.clamp(x2_inter - x1_inter, 0) * torch.clamp(y2_inter - y1_inter, 0)
        
        # Calculate individual box areas
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        
        # Calculate Dice coefficient
        dice = (2.0 * inter_area + self.smooth) / (pred_area + target_area + self.smooth)
        
        # Return DSC loss (1 - dice)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss function using both standard YOLO loss and DSC loss.
    """
    
    def __init__(self, dsc_weight=0.3, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.dsc_weight = dsc_weight
        self.dsc_loss = DSCLoss(smooth)
        self.bbox_loss = BboxLoss(reg_max=16)  # Standard YOLO bbox loss
    
    def forward(self, pred_boxes, target_boxes, **kwargs):
        """
        Calculate combined loss.
        
        Args:
            pred_boxes: Predicted bounding boxes
            target_boxes: Target bounding boxes
        
        Returns:
            Combined loss value
        """
        # Standard bbox loss
        standard_loss = self.bbox_loss(pred_boxes, target_boxes, **kwargs)
        
        # DSC loss for better IoU
        dsc_loss = self.dsc_loss(pred_boxes, target_boxes)
        
        # Combine losses
        total_loss = (1 - self.dsc_weight) * standard_loss + self.dsc_weight * dsc_loss
        
        return total_loss


class CustomYOLO(YOLO):
    """
    Custom YOLO class with DSC loss integration.
    """
    
    def __init__(self, model='yolov8x.pt', dsc_weight=0.3):
        super().__init__(model)
        self.dsc_weight = dsc_weight
        self.custom_loss = None
    
    def _setup_loss(self):
        """Setup custom loss function with DSC component."""
        if hasattr(self.model, 'model') and hasattr(self.model.model[-1], 'loss'):
            # Get the existing loss function
            original_loss = self.model.model[-1].loss
            
            # Create combined loss
            self.custom_loss = CombinedLoss(dsc_weight=self.dsc_weight)
            
            # Replace the loss function
            self.model.model[-1].loss = self.custom_loss
            print(f"✅ Custom DSC loss initialized with weight: {self.dsc_weight}")
        else:
            print("⚠️  Could not setup custom loss - using default YOLO loss")


def train_with_dsc_loss():
    """Train model with DSC loss for improved bounding box accuracy."""
    print("🚀 Starting Snap Circuit Training with DSC Loss...")
    print("="*60)
    
    # Check for augmented dataset
    augmented_data_yaml = "data/augmented_training/data.yaml"
    
    if not os.path.exists(augmented_data_yaml):
        print("❌ Augmented dataset not found!")
        print("   Please ensure the dataset exists at:", augmented_data_yaml)
        return
    
    # Initialize custom YOLO with DSC loss
    print("🔧 Initializing Custom YOLO with DSC Loss...")
    model = CustomYOLO('yolov8x.pt', dsc_weight=0.3)
    
    # Setup custom loss after model initialization
    # Note: This will be handled during training initialization
    
    # Enhanced training parameters optimized for DSC loss
    training_params = {
        "epochs": 150,              # Adjusted for DSC loss convergence
        "imgsz": 640,              # Standard size
        "batch": 12,               # Slightly reduced for memory efficiency
        "lr0": 0.0005,             # Lower learning rate for stable training
        "lrf": 0.01,               # Final learning rate factor
        "patience": 40,            # Patience for early stopping
        "device": "cuda",          # GPU training
        "optimizer": "AdamW",      # Adam with weight decay
        "cos_lr": True,            # Cosine learning rate scheduler
        "warmup_epochs": 5,        # Warmup epochs
        "warmup_momentum": 0.8,    # Warmup momentum
        "weight_decay": 0.0005,    # Weight decay for regularization
        
        # Data augmentation (moderate for better DSC loss performance)
        "augment": True,
        "mixup": 0.05,             # Reduced mixup
        "mosaic": 0.3,             # Reduced mosaic
        "copy_paste": 0.05,        # Reduced copy-paste
        "hsv_h": 0.015,            # HSV hue augmentation
        "hsv_s": 0.7,              # HSV saturation
        "hsv_v": 0.4,              # HSV value
        "degrees": 10.0,           # Rotation degrees
        "translate": 0.1,          # Translation
        "scale": 0.5,              # Scale variation
        "shear": 2.0,              # Shear
        "flipud": 0.0,             # No vertical flip
        "fliplr": 0.5,             # Horizontal flip
        
        # Training settings
        "save_period": 10,         # Save every 10 epochs
        "val": True,               # Enable validation
        "plots": True,             # Generate training plots
        "verbose": True,           # Verbose output
        "exist_ok": True,          # Allow overwriting existing runs
        "project": "snap_circuit_training",
        "name": "dsc_loss_model"
    }
    
    print("📊 DSC Loss Training Parameters:")
    print(f"   📁 Dataset: {augmented_data_yaml}")
    print(f"   🎯 DSC Loss Weight: {model.dsc_weight}")
    print(f"   📈 Epochs: {training_params['epochs']}")
    print(f"   📦 Batch Size: {training_params['batch']}")
    print(f"   🎓 Learning Rate: {training_params['lr0']}")
    print(f"   🔄 Optimizer: {training_params['optimizer']}")
    
    try:
        # Train the model with DSC loss
        print("\n🏋️ Starting DSC Loss Training...")
        print("   This will focus on improving bounding box accuracy...")
        
        results = model.train(
            data=augmented_data_yaml,
            **training_params
        )
        
        print("✅ DSC Loss Training Complete!")
        print(f"📁 Results saved to: {results.save_dir}")
        
        # Validate the model
        print("\n📊 Running Validation with DSC Loss Model...")
        val_results = model.val(data=augmented_data_yaml, imgsz=640)
        
        print("✅ Validation Complete!")
        
        # Extract and display metrics
        if hasattr(val_results, 'box'):
            metrics = val_results.box
            print(f"\n📈 Validation Metrics:")
            print(f"   📊 mAP50: {metrics.map50:.4f}")
            print(f"   📊 mAP50-95: {metrics.map:.4f}")
            print(f"   🎯 Precision: {metrics.mp:.4f}")
            print(f"   🔍 Recall: {metrics.mr:.4f}")
        
        # Copy best model to our models directory
        best_model_path = results.save_dir / "weights" / "best.pt"
        if best_model_path.exists():
            import shutil
            models_dir = Path("models/weights")
            models_dir.mkdir(parents=True, exist_ok=True)
            
            target_path = models_dir / "snap_circuit_dsc_loss.pt"
            shutil.copy(best_model_path, target_path)
            print(f"📁 Best model copied to: {target_path}")
        
        # Test the model with evaluation script
        print("\n🔍 Running Evaluation on DSC Loss Model...")
        test_model_path = str(best_model_path) if best_model_path.exists() else None
        
        if test_model_path:
            # Import and run evaluation
            from evaluate_model import ModelEvaluator
            
            evaluator = ModelEvaluator(test_model_path, augmented_data_yaml)
            metrics = evaluator.evaluate_dataset('val')
            evaluator.print_results(metrics)
            
            # Save DSC loss results
            dsc_results_path = "output/dsc_loss_evaluation_results.json"
            evaluator.save_results(metrics, dsc_results_path)
            
            print(f"\n📊 DSC Loss Model Evaluation:")
            print(f"   📈 RMSE: {metrics.rmse_pixels:.2f} pixels")
            print(f"   📈 MAE: {metrics.mae_pixels:.2f} pixels")
            print(f"   🎯 Precision: {metrics.precision:.3f}")
            print(f"   🔍 Recall: {metrics.recall:.3f}")
            print(f"   📁 Results saved to: {dsc_results_path}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during DSC loss training: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_models():
    """Compare DSC loss model with original model."""
    print("\n🔍 Comparing DSC Loss Model vs Original Model...")
    print("="*50)
    
    # Paths
    dsc_results_path = "output/dsc_loss_evaluation_results.json"
    original_results_path = "output/evaluation_results.json"
    
    if os.path.exists(dsc_results_path) and os.path.exists(original_results_path):
        import json
        
        # Load results
        with open(dsc_results_path, 'r') as f:
            dsc_results = json.load(f)
        
        with open(original_results_path, 'r') as f:
            original_results = json.load(f)
        
        print("📊 Model Comparison:")
        print(f"   {'Metric':<15} {'Original':<12} {'DSC Loss':<12} {'Improvement':<12}")
        print("-" * 55)
        
        # Compare key metrics
        for metric in ['rmse_pixels', 'mae_pixels']:
            orig_val = original_results['overall_metrics'][metric]
            dsc_val = dsc_results['overall_metrics'][metric]
            improvement = ((orig_val - dsc_val) / orig_val * 100) if orig_val > 0 else 0
            
            print(f"   {metric.replace('_', ' ').title():<15} {orig_val:<12.2f} {dsc_val:<12.2f} {improvement:<12.1f}%")
        
        for metric in ['precision', 'recall', 'f1_score']:
            orig_val = original_results['overall_metrics'][metric]
            dsc_val = dsc_results['overall_metrics'][metric]
            improvement = ((dsc_val - orig_val) / orig_val * 100) if orig_val > 0 else 0
            
            print(f"   {metric.replace('_', ' ').title():<15} {orig_val:<12.3f} {dsc_val:<12.3f} {improvement:<12.1f}%")
    
    else:
        print("⚠️  Comparison results not available - run both evaluations first")


if __name__ == "__main__":
    # Train with DSC loss
    results = train_with_dsc_loss()
    
    if results:
        print("\n🎉 DSC Loss Training Pipeline Complete!")
        
        # Compare models if both results exist
        compare_models()
    else:
        print("❌ Training failed - please check the error messages above") 