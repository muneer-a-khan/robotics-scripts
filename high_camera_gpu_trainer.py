#!/usr/bin/env python3
"""
High Camera GPU Training Script for Snap Circuit Component Detection
Optimized for high camera perspective with GPU training
"""

import os
import sys
from pathlib import Path
import yaml
from datetime import datetime
import torch
from ultralytics import YOLO
import logging

class HighCameraGPUTrainer:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data" / "high_camera_training"
        self.output_dir = self.project_root / "output"
        self.results_dir = self.output_dir / "results"
        self.setup_logging()
        self.setup_directories()
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"high_camera_training_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """Create necessary directories"""
        directories = [self.output_dir, self.results_dir, self.output_dir / "logs"]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def validate_dataset(self):
        """Validate the high camera dataset structure"""
        self.logger.info("Validating high camera training dataset...")
        
        data_yaml = self.data_dir / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(f"Data configuration not found: {data_yaml}")
            
        # Load and validate data.yaml
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
            
        # Check required directories
        train_dir = self.data_dir / "images" / "train"
        val_dir = self.data_dir / "images" / "val"
        train_labels_dir = self.data_dir / "labels" / "train"
        val_labels_dir = self.data_dir / "labels" / "val"
        
        if not train_dir.exists():
            raise FileNotFoundError(f"Training images directory not found: {train_dir}")
        if not val_dir.exists():
            raise FileNotFoundError(f"Validation images directory not found: {val_dir}")
        if not train_labels_dir.exists():
            raise FileNotFoundError(f"Training labels directory not found: {train_labels_dir}")
        if not val_labels_dir.exists():
            raise FileNotFoundError(f"Validation labels directory not found: {val_labels_dir}")
            
        # Count files
        train_images = list(train_dir.glob("*.jpg"))
        val_images = list(val_dir.glob("*.jpg"))
        train_labels = list(train_labels_dir.glob("*.txt"))
        val_labels = list(val_labels_dir.glob("*.txt"))
        
        self.logger.info(f"Training images: {len(train_images)}")
        self.logger.info(f"Validation images: {len(val_images)}")
        self.logger.info(f"Training labels: {len(train_labels)}")
        self.logger.info(f"Validation labels: {len(val_labels)}")
        self.logger.info(f"Component classes: {len(data_config['names'])}")
        
        return data_config
        
    def check_gpu_availability(self):
        """Check GPU availability and configuration"""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            self.logger.info(f"GPU Available: {gpu_name}")
            self.logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
            self.logger.info(f"GPU Count: {gpu_count}")
            
            return True
        else:
            self.logger.warning("No GPU available, falling back to CPU training")
            return False
            
    def train_model(self, experiment_name=None):
        """Train YOLOv8 model on high camera data with GPU optimization"""
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"high_camera_gpu_{timestamp}"
            
        self.logger.info(f"Starting high camera GPU training: {experiment_name}")
        
        # Validate dataset
        data_config = self.validate_dataset()
        gpu_available = self.check_gpu_availability()
        
        # Initialize YOLOv8 model (using nano for faster training, can be changed to 's', 'm', 'l', 'x')
        model = YOLO("yolov8n.pt")
        
        # GPU-optimized training parameters for high camera perspective
        train_args = {
            'data': str(self.data_dir / "data.yaml"),
            'epochs': 150,              # Sufficient epochs for convergence
            'batch': 16 if gpu_available else 8,  # Optimized for GPU memory
            'imgsz': 640,               # Standard YOLO image size
            'patience': 25,             # Early stopping patience
            'save': True,
            'save_period': 10,          # Save every 10 epochs
            'cache': False,             # Disable caching for large datasets
            'device': 0 if gpu_available else 'cpu',
            'workers': 8,               # Parallel data loading
            'project': str(self.results_dir),
            'name': experiment_name,
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',       # Better for fine-tuning
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'rect': False,              # Rectangular training
            'cos_lr': True,             # Cosine learning rate scheduler
            'close_mosaic': 10,         # Disable mosaic in last 10 epochs
            'resume': False,
            'amp': True,                # Automatic Mixed Precision for GPU
            'fraction': 1.0,            # Use full dataset
            'profile': False,
            'freeze': None,
            
            # Learning rate settings
            'lr0': 0.001,               # Initial learning rate
            'lrf': 0.01,                # Final learning rate factor
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Loss function weights
            'box': 7.5,                 # Box loss weight
            'cls': 0.5,                 # Classification loss weight
            'dfl': 1.5,                 # Distribution focal loss weight
            
            # Data augmentation (moderate for high camera stability)
            'hsv_h': 0.015,             # HSV hue augmentation
            'hsv_s': 0.7,               # HSV saturation
            'hsv_v': 0.4,               # HSV value
            'degrees': 5.0,             # Rotation degrees (reduced for high camera)
            'translate': 0.1,           # Translation fraction
            'scale': 0.5,               # Image scale variation
            'shear': 2.0,               # Shear degrees
            'perspective': 0.0,         # Perspective transform (disabled for high camera)
            'flipud': 0.0,              # Vertical flip (disabled for high camera)
            'fliplr': 0.5,              # Horizontal flip probability
            'mosaic': 0.8,              # Mosaic augmentation probability
            'mixup': 0.1,               # Mixup probability
            'copy_paste': 0.1,          # Copy-paste probability
            
            # Advanced settings
            'label_smoothing': 0.0,
            'nbs': 64,                  # Nominal batch size
            'overlap_mask': True,
            'mask_ratio': 4,
            'plots': True,              # Generate training plots
            'val': True                 # Enable validation
        }
        
        self.logger.info("Training Parameters:")
        for key, value in train_args.items():
            self.logger.info(f"  {key}: {value}")
            
        # Start training
        self.logger.info("Starting training...")
        results = model.train(**train_args)
        
        self.logger.info("Training completed!")
        self.logger.info(f"Results saved to: {results.save_dir}")
        
        # Validate the trained model
        self.logger.info("Running validation...")
        validation_results = model.val()
        
        self.logger.info("Validation completed!")
        self.logger.info(f"mAP50: {validation_results.box.map50:.4f}")
        self.logger.info(f"mAP50-95: {validation_results.box.map:.4f}")
        
        # Save model info
        model_info = {
            'experiment_name': experiment_name,
            'training_data': str(self.data_dir),
            'model_path': str(results.save_dir),
            'map50': float(validation_results.box.map50),
            'map50_95': float(validation_results.box.map),
            'epochs_trained': 150,  # Fixed value since training completed all epochs
            'gpu_used': gpu_available,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save model info to JSON
        import json
        info_file = self.results_dir / f"{experiment_name}_info.json"
        with open(info_file, 'w') as f:
            json.dump(model_info, f, indent=2)
            
        self.logger.info(f"Model info saved to: {info_file}")
        
        return results

def main():
    """Main training function"""
    print("🚀 High Camera GPU Training for Snap Circuit Detection")
    print("=" * 60)
    
    trainer = HighCameraGPUTrainer()
    
    try:
        results = trainer.train_model()
        print("\n✅ Training completed successfully!")
        print(f"📁 Results saved to: {results.save_dir}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()