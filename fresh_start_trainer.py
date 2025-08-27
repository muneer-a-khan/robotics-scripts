"""
Fresh Start Trainer for Snap Circuit Component Detection
Complete restart with modern YOLOv8 training pipeline
"""

import os
import sys
from pathlib import Path
import yaml
from datetime import datetime
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from fresh_start_config import FreshStartConfig

class FreshStartTrainer:
    def __init__(self):
        self.config = FreshStartConfig()
        self.setup_logging()
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = self.config.OUTPUT_DIR / "logs"
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"training_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def validate_dataset(self):
        """Validate the dataset structure and annotations"""
        self.logger.info("Validating dataset...")
        
        data_yaml = self.config.get_data_yaml_path()
        if not data_yaml.exists():
            raise FileNotFoundError(f"data.yaml not found at {data_yaml}")
            
        # Load and validate data.yaml
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
            
        # Check paths
        base_path = data_yaml.parent
        for split in ['train', 'val', 'test']:
            if split in data_config:
                split_path = base_path / data_config[split]
                if not split_path.exists():
                    self.logger.warning(f"Split path {split_path} does not exist")
                    
        self.logger.info(f"Dataset validation complete. Classes: {data_config.get('nc', 'Unknown')}")
        return data_config
        
    def train_model(self, experiment_name=None):
        """Train the YOLOv8 model with fresh configuration"""
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"fresh_start_{timestamp}"
            
        self.logger.info(f"Starting fresh training experiment: {experiment_name}")
        
        # Validate dataset first
        data_config = self.validate_dataset()
        
        # Initialize YOLOv8 model
        model = YOLO(f"yolov8{self.config.MODEL_SIZE}.pt")
        
        # Training arguments
        train_args = {
            'data': str(self.config.get_data_yaml_path()),
            'epochs': self.config.EPOCHS,
            'batch': self.config.BATCH_SIZE,
            'imgsz': self.config.IMAGE_SIZE,
            'patience': self.config.PATIENCE,
            'save': True,
            'save_period': 10,
            'cache': False,
            'device': 0 if torch.cuda.is_available() else 'cpu',
            'workers': 8,
            'project': str(self.config.RESULTS_DIR),
            'name': experiment_name,
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'auto',
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': True,
            'close_mosaic': 10,
            'resume': False,
            'amp': True,
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'plots': True,
        }
        
        # Log GPU information
        if torch.cuda.is_available():
            self.logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        else:
            self.logger.info("CUDA not available, using CPU")
            
        self.logger.info("Starting model training...")
        self.logger.info(f"Training arguments: {train_args}")
        
        # Start training
        try:
            results = model.train(**train_args)
            
            # Save the trained model
            model_save_path = self.config.get_model_save_path(experiment_name)
            model.export(format='torchscript')
            model.save(model_save_path)
            
            self.logger.info(f"Training completed successfully!")
            self.logger.info(f"Model saved to: {model_save_path}")
            
            return results, model
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
            
    def validate_model(self, model_path):
        """Validate the trained model on validation set"""
        self.logger.info("Validating trained model...")
        
        model = YOLO(model_path)
        
        # Run validation
        val_results = model.val(
            data=str(self.config.get_data_yaml_path()),
            split='val',
            imgsz=self.config.IMAGE_SIZE,
            batch=self.config.BATCH_SIZE,
            plots=True,
            save_json=True,
            save_hybrid=True,
            conf=self.config.CONFIDENCE_THRESHOLD,
            iou=0.5,
            max_det=self.config.MAX_DETECTIONS,
            half=True,
            dnn=False
        )
        
        self.logger.info(f"Validation completed. Results: {val_results}")
        return val_results

def main():
    """Main training function"""
    trainer = FreshStartTrainer()
    
    try:
        # Train the model
        results, model = trainer.train_model()
        
        # Validate the model
        model_path = trainer.config.get_model_save_path()
        if model_path.exists():
            trainer.validate_model(model_path)
        else:
            trainer.logger.warning("Model file not found for validation")
            
    except Exception as e:
        trainer.logger.error(f"Training process failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 