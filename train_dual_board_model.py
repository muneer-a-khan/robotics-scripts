#!/usr/bin/env python3
"""
Dual Board Model Training Script

This script trains a YOLOv8/YOLOv11 model specifically for dual circuit board detection.
It handles the augmented dataset created by the dual board pipeline and includes
specialized training parameters for circuit component detection.

Features:
- Optimized training parameters for circuit detection
- GPU acceleration with memory management
- Early stopping and learning rate scheduling
- Comprehensive training monitoring
- Model validation and metrics calculation
- Integration with existing pipeline components
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
import torch
from ultralytics import YOLO
import yaml

from config import get_optimal_device
from models.component_detector import ComponentDetector
from early_stopping_controller import EarlyStoppingController


class DualBoardModelTrainer:
    """Advanced trainer for dual board circuit detection models"""
    
    def __init__(self, 
                 base_model: str = "yolov8x.pt",
                 experiment_name: str = None,
                 use_gpu: bool = True):
        """
        Initialize the dual board trainer
        
        Args:
            base_model: Base YOLO model to start from
            experiment_name: Name for this training experiment
            use_gpu: Whether to use GPU acceleration
        """
        self.project_root = Path(__file__).parent
        self.base_model = base_model
        self.experiment_name = experiment_name or f"dual_board_{int(time.time())}"
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Setup logging
        self.setup_logging()
        
        # Training state
        self.model = None
        self.data_yaml_path = None
        self.output_dir = None
        
        # Early stopping
        self.early_stopping = EarlyStoppingController(
            patience=30,
            min_delta=0.001,
            monitor_metric='mAP50'
        )
        
        self.logger.info(f"Dual Board Model Trainer Initialized")
        self.logger.info(f"   Base model: {self.base_model}")
        self.logger.info(f"   Experiment: {self.experiment_name}")
        self.logger.info(f"   GPU available: {torch.cuda.is_available()}")
        self.logger.info(f"   Using GPU: {self.use_gpu}")
        
        if self.use_gpu:
            gpu_info = self.get_gpu_info()
            self.logger.info(f"   GPU: {gpu_info}")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"dual_board_training_{self.experiment_name}.log"
        
        # Create logger
        self.logger = logging.getLogger(f"DualBoardTrainer_{self.experiment_name}")
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging setup complete: {log_file}")
    
    def get_gpu_info(self) -> str:
        """Get GPU information"""
        if not torch.cuda.is_available():
            return "No GPU available"
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return f"{gpu_name} ({gpu_memory:.1f}GB)"
    
    def validate_dataset(self, data_yaml_path: str) -> bool:
        """
        Validate the dataset before training
        
        Args:
            data_yaml_path: Path to data.yaml file
            
        Returns:
            True if dataset is valid
        """
        self.logger.info("Validating dataset...")
        
        try:
            # Load data.yaml
            with open(data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
            
            required_keys = ['path', 'train', 'val', 'nc', 'names']
            for key in required_keys:
                if key not in data_config:
                    self.logger.error(f"Missing required key in data.yaml: {key}")
                    return False
            
            # Check paths
            dataset_root = Path(data_config['path'])
            train_path = dataset_root / data_config['train']
            val_path = dataset_root / data_config['val']
            
            if not dataset_root.exists():
                self.logger.error(f"Dataset root does not exist: {dataset_root}")
                return False
            
            if not train_path.exists():
                self.logger.error(f"Training path does not exist: {train_path}")
                return False
            
            if not val_path.exists():
                self.logger.error(f"Validation path does not exist: {val_path}")
                return False
            
            # Count images
            train_images = len(list(train_path.glob('*.jpg'))) + len(list(train_path.glob('*.png')))
            val_images = len(list(val_path.glob('*.jpg'))) + len(list(val_path.glob('*.png')))
            
            # Count labels
            labels_train_path = dataset_root / "labels" / "train"
            labels_val_path = dataset_root / "labels" / "val"
            
            train_labels = len(list(labels_train_path.glob('*.txt'))) if labels_train_path.exists() else 0
            val_labels = len(list(labels_val_path.glob('*.txt'))) if labels_val_path.exists() else 0
            
            self.logger.info(f"   Dataset statistics:")
            self.logger.info(f"     Classes: {data_config['nc']}")
            self.logger.info(f"     Training images: {train_images}")
            self.logger.info(f"     Training labels: {train_labels}")
            self.logger.info(f"     Validation images: {val_images}")
            self.logger.info(f"     Validation labels: {val_labels}")
            
            if train_images == 0:
                self.logger.error("No training images found!")
                return False
            
            if val_images == 0:
                self.logger.warning("No validation images found - will use training data for validation")
            
            if train_labels < train_images * 0.5:
                self.logger.warning(f"Many images may be missing labels ({train_labels}/{train_images})")
            
            self.logger.info("Dataset validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Dataset validation failed: {e}")
            return False
    
    def calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available GPU memory"""
        if not self.use_gpu:
            return 4  # Conservative CPU batch size
        
        try:
            # Get GPU memory in GB
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # Rough estimation based on YOLOv8 memory usage
            if gpu_memory_gb >= 24:      # RTX 4090, A100
                batch_size = 32
            elif gpu_memory_gb >= 16:    # RTX 4080, V100
                batch_size = 24
            elif gpu_memory_gb >= 12:    # RTX 4070 Ti, RTX 3080 Ti
                batch_size = 16
            elif gpu_memory_gb >= 8:     # RTX 4060 Ti, RTX 3070
                batch_size = 12
            elif gpu_memory_gb >= 6:     # RTX 3060
                batch_size = 8
            else:                        # RTX 3050 and below
                batch_size = 4
            
            self.logger.info(f"Calculated optimal batch size: {batch_size} (GPU: {gpu_memory_gb:.1f}GB)")
            return batch_size
            
        except Exception as e:
            self.logger.warning(f"Could not calculate optimal batch size: {e}")
            return 8  # Safe default
    
    def get_training_parameters(self, epochs: int = 200, 
                              imgsz: int = 640,
                              batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Get optimized training parameters for dual board detection
        
        Args:
            epochs: Number of training epochs
            imgsz: Input image size
            batch_size: Batch size (auto-calculated if None)
            
        Returns:
            Dictionary of training parameters
        """
        if batch_size is None:
            batch_size = self.calculate_optimal_batch_size()
        
        # Device configuration
        device = get_optimal_device() if self.use_gpu else "cpu"
        
        params = {
            # Basic training parameters
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'device': device,
            'workers': 8 if self.use_gpu else 2,
            'project': 'dual_board_training',
            'name': self.experiment_name,
            'exist_ok': True,
            
            # Optimizer settings
            'optimizer': 'AdamW',           # Better for fine-tuning
            'lr0': 0.001,                   # Initial learning rate
            'lrf': 0.01,                    # Final learning rate (lr0 * lrf)
            'momentum': 0.937,              # SGD momentum
            'weight_decay': 0.0005,         # Weight decay
            'warmup_epochs': 5,             # Warmup epochs
            'warmup_momentum': 0.8,         # Warmup momentum
            'warmup_bias_lr': 0.1,          # Warmup bias learning rate
            
            # Learning rate schedule
            'cos_lr': True,                 # Cosine learning rate schedule
            'patience': 30,                 # Early stopping patience
            'close_mosaic': 15,             # Close mosaic augmentation last N epochs
            
            # Loss function weights (optimized for circuit detection)
            'box': 7.5,                     # Box loss weight
            'cls': 0.5,                     # Classification loss weight  
            'dfl': 1.5,                     # Distribution focal loss weight
            
            # Data augmentation (moderate for circuit stability)
            'hsv_h': 0.015,                 # HSV hue augmentation (range: 0-1)
            'hsv_s': 0.5,                   # HSV saturation augmentation
            'hsv_v': 0.4,                   # HSV value augmentation
            'degrees': 5.0,                 # Rotation degrees (small for circuits)
            'translate': 0.1,               # Translation fraction
            'scale': 0.3,                   # Image scale (+/-)
            'shear': 2.0,                   # Shear degrees
            'perspective': 0.0,             # Perspective transform (disabled for top-down)
            'flipud': 0.0,                  # Vertical flip (disabled - circuits have orientation)
            'fliplr': 0.5,                  # Horizontal flip probability
            'mosaic': 0.3,                  # Mosaic augmentation probability
            'mixup': 0.05,                  # Mixup probability (low for circuits)
            'copy_paste': 0.05,             # Copy-paste probability
            
            # Validation and saving
            'val': True,                    # Validate during training
            'save': True,                   # Save model
            'save_period': 10,              # Save every N epochs
            'plots': True,                  # Generate training plots
            'verbose': True,                # Verbose output
            
            # Advanced settings
            'amp': True,                    # Automatic mixed precision
            'fraction': 1.0,                # Dataset fraction to use
            'profile': False,               # Profile ONNX and TensorRT speeds
            'freeze': None,                 # Freeze layers: backbone=10, first3=0 1 2
            'multi_scale': False,           # Multi-scale training
            'overlap_mask': True,           # Overlap masks
            'mask_ratio': 4,                # Mask downsample ratio
            'dropout': 0.0,                 # Use dropout regularization
            'label_smoothing': 0.0,         # Label smoothing epsilon
            'nbs': 64,                      # Nominal batch size
        }
        
        return params
    
    def train_model(self, data_yaml_path: str, **kwargs) -> Optional[str]:
        """
        Train the dual board detection model
        
        Args:
            data_yaml_path: Path to dataset configuration
            **kwargs: Additional training parameters
            
        Returns:
            Path to trained model or None if failed
        """
        self.logger.info("Starting dual board model training...")
        
        # Validate dataset
        if not self.validate_dataset(data_yaml_path):
            self.logger.error("Dataset validation failed!")
            return None
        
        self.data_yaml_path = data_yaml_path
        
        try:
            # Load base model
            self.logger.info(f"📥 Loading base model: {self.base_model}")
            self.model = YOLO(self.base_model)
            
            # Get training parameters
            train_params = self.get_training_parameters()
            train_params.update(kwargs)  # Override with any custom parameters
            
            self.logger.info("Training Parameters:")
            for key, value in train_params.items():
                self.logger.info(f"   {key}: {value}")
            
            # Start training
            self.logger.info("Beginning training process...")
            start_time = time.time()
            
            results = self.model.train(data=data_yaml_path, **train_params)
            
            training_time = time.time() - start_time
            self.logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Get trained model path
            if hasattr(results, 'save_dir'):
                model_path = Path(results.save_dir) / "weights" / "best.pt"
                if model_path.exists():
                    self.logger.info(f"🎉 Best model saved to: {model_path}")
                    
                    # Copy to standard location
                    standard_path = self.project_root / "models" / "weights" / f"dual_board_{self.experiment_name}.pt"
                    standard_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    import shutil
                    shutil.copy2(model_path, standard_path)
                    self.logger.info(f"📋 Model copied to: {standard_path}")
                    
                    # Save training summary
                    self.save_training_summary(results, training_time)
                    
                    return str(standard_path)
                else:
                    self.logger.error(f"Best model not found at: {model_path}")
                    return None
            else:
                self.logger.error("Training results do not contain save directory")
                return None
                
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def save_training_summary(self, results, training_time: float):
        """Save comprehensive training summary"""
        summary_dir = self.project_root / "training_summaries"
        summary_dir.mkdir(exist_ok=True)
        
        summary_file = summary_dir / f"dual_board_{self.experiment_name}_summary.json"
        
        summary = {
            "experiment_name": self.experiment_name,
            "base_model": self.base_model,
            "data_yaml_path": str(self.data_yaml_path),
            "training_time_seconds": training_time,
            "training_time_formatted": f"{training_time/60:.1f} minutes",
            "gpu_used": self.use_gpu,
            "gpu_info": self.get_gpu_info() if self.use_gpu else None,
            "timestamp": time.time(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # Add results info if available
        if hasattr(results, 'save_dir'):
            summary["results_dir"] = str(results.save_dir)
        
        # Save training parameters
        train_params = self.get_training_parameters()
        summary["training_parameters"] = train_params
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"📄 Training summary saved: {summary_file}")
    
    def validate_model(self, model_path: str, data_yaml_path: str) -> Dict[str, Any]:
        """
        Run comprehensive validation on trained model
        
        Args:
            model_path: Path to trained model
            data_yaml_path: Path to dataset configuration
            
        Returns:
            Validation metrics
        """
        self.logger.info("🧪 Running model validation...")
        
        try:
            # Load trained model
            model = YOLO(model_path)
            
            # Run validation
            val_results = model.val(data=data_yaml_path, imgsz=640, batch=1)
            
            # Extract key metrics
            metrics = {
                "map50": float(val_results.box.map50) if hasattr(val_results.box, 'map50') else None,
                "map50_95": float(val_results.box.map) if hasattr(val_results.box, 'map') else None,
                "precision": float(val_results.box.mp) if hasattr(val_results.box, 'mp') else None,
                "recall": float(val_results.box.mr) if hasattr(val_results.box, 'mr') else None,
            }
            
            self.logger.info("Validation Results:")
            for metric, value in metrics.items():
                if value is not None:
                    self.logger.info(f"   {metric}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return {}
    
    def create_training_script(self, data_yaml_path: str, output_script: str = "run_dual_board_training.py"):
        """Create a standalone training script"""
        script_content = f'''#!/usr/bin/env python3
"""
Auto-generated Dual Board Training Script
Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""

from train_dual_board_model import DualBoardModelTrainer

def main():
    # Initialize trainer
    trainer = DualBoardModelTrainer(
        base_model="yolov8x.pt",
        experiment_name="dual_board_auto_{int(time.time())}",
        use_gpu=True
    )
    
    # Train model
    model_path = trainer.train_model(
        data_yaml_path="{data_yaml_path}",
        epochs=200,
        imgsz=640
    )
    
    if model_path:
        print(f"✅ Training completed! Model saved to: {{model_path}}")
        
        # Run validation
        metrics = trainer.validate_model(model_path, "{data_yaml_path}")
        print(f"📊 Validation metrics: {{metrics}}")
    else:
        print("❌ Training failed!")

if __name__ == "__main__":
    main()
'''
        
        script_path = Path(output_script)
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable on Unix systems
        try:
            os.chmod(script_path, 0o755)
        except:
            pass
        
        self.logger.info(f"📜 Training script created: {script_path}")
        return str(script_path)


def main():
    """Main entry point for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dual Board Model Trainer")
    parser.add_argument("--data", "-d", required=True, help="Path to data.yaml file")
    parser.add_argument("--base-model", "-m", default="yolov8x.pt", help="Base YOLO model")
    parser.add_argument("--name", "-n", help="Experiment name")
    parser.add_argument("--epochs", "-e", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch", "-b", type=int, help="Batch size (auto if not specified)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("--validate", action="store_true", help="Run validation after training")
    parser.add_argument("--create-script", action="store_true", help="Create standalone training script")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = DualBoardModelTrainer(
        base_model=args.base_model,
        experiment_name=args.name,
        use_gpu=not args.cpu
    )
    
    # Create training script if requested
    if args.create_script:
        script_path = trainer.create_training_script(args.data)
        print(f"📜 Training script created: {script_path}")
        print(f"🚀 Run with: python {script_path}")
        return
    
    # Train model
    model_path = trainer.train_model(
        data_yaml_path=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch
    )
    
    if model_path:
        print(f"✅ Training completed! Model saved to: {model_path}")
        
        # Run validation if requested
        if args.validate:
            metrics = trainer.validate_model(model_path, args.data)
            print(f"📊 Validation metrics: {metrics}")
    else:
        print("❌ Training failed!")


if __name__ == "__main__":
    main()
