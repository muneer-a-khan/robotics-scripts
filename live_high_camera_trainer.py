#!/usr/bin/env python3
"""
Live High Camera Training System

This system combines real-time dual camera processing with continuous model retraining
for optimal high-up camera performance. It captures new data, augments it, and 
retrains the model periodically.

Features:
- Real-time dual camera processing
- Automatic data collection from live camera feed
- Periodic model retraining with new data
- High-camera specific augmentations (zoom, resolution variants)
- Seamless model switching during operation

Usage:
    # Start live training with dual camera
    python live_high_camera_trainer.py
    
    # Custom training intervals
    python live_high_camera_trainer.py --retrain-interval 300  # 5 minutes
    
    # Collect data only (no retraining)
    python live_high_camera_trainer.py --collect-only
"""

import os
import sys
import time
import json
import shutil
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import subprocess
import argparse
from typing import Optional, Dict, List
import cv2
import numpy as np
from ultralytics import YOLO

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from high_camera_training_prep import HighCameraTrainingPrep
from dual_camera_system import DualCircuitBoardSystem


class LiveHighCameraTrainer:
    """
    Live training system for high camera mode with dual circuit board processing.
    """
    
    def __init__(self,
                 retrain_interval: int = 600,  # 10 minutes
                 min_new_samples: int = 50,
                 max_training_data: int = 2000,
                 collect_only: bool = False):
        """
        Initialize live high camera trainer.
        
        Args:
            retrain_interval: Seconds between retraining attempts
            min_new_samples: Minimum new samples before retraining
            max_training_data: Maximum training samples to keep
            collect_only: Only collect data, don't retrain
        """
        self.retrain_interval = retrain_interval
        self.min_new_samples = min_new_samples
        self.max_training_data = max_training_data
        self.collect_only = collect_only
        
        # Initialize paths
        self.live_data_dir = Path("data/live_high_camera_training")
        self.live_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.live_data_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.live_data_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (self.live_data_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (self.live_data_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        # Training preparation system
        self.training_prep = HighCameraTrainingPrep()
        
        # Current model path (starts with existing model)
        self.current_model_path = self._find_best_model()
        
        # Dual camera system
        self.dual_system = None
        
        # Training state
        self.samples_collected = 0
        self.last_retrain_time = time.time()
        self.training_generation = 0
        self.is_training = False
        
        # Threading
        self.stop_event = threading.Event()
        self.data_collection_thread = None
        self.training_thread = None
        
        print(f"🎯 Live High Camera Trainer Initialized")
        print(f"   📊 Retrain interval: {retrain_interval}s")
        print(f"   🔢 Min samples for retrain: {min_new_samples}")
        print(f"   📁 Live data directory: {self.live_data_dir}")
        print(f"   🤖 Current model: {self.current_model_path}")
        print(f"   🎮 Collect only mode: {collect_only}")
    
    def _find_best_model(self) -> str:
        """Find the best available model to start with."""
        model_candidates = [
            "high_camera_training/high_camera_model2/weights/best.pt",
            "high_camera_training/high_camera_model/weights/best.pt",
            "latest_trained_model.pt",
            "models/weights/latest_trained_model.pt"
        ]
        
        for model_path in model_candidates:
            if Path(model_path).exists():
                print(f"✅ Found starting model: {model_path}")
                return model_path
        
        # If no model found, use default YOLOv8
        print("⚠️  No trained model found, using YOLOv8x pretrained")
        return "yolov8x.pt"
    
    def start_data_collection(self, camera_id: int = 0):
        """Start collecting training data from dual camera system."""
        print("🎥 Starting live data collection...")
        
        # Initialize dual camera system with current model
        self.dual_system = DualCircuitBoardSystem(
            model_path=self.current_model_path,
            save_outputs=True,
            display_results=True,
            enable_validation=True
        )
        
        # Start data collection thread
        self.data_collection_thread = threading.Thread(
            target=self._collect_data_worker,
            args=(camera_id,)
        )
        self.data_collection_thread.start()
        
        if not self.collect_only:
            # Start training thread
            self.training_thread = threading.Thread(
                target=self._training_worker
            )
            self.training_thread.start()
    
    def _collect_data_worker(self, camera_id: int):
        """Worker thread for collecting training data."""
        print(f"📸 Data collection worker started (Camera {camera_id})")
        
        try:
            # Run dual camera system
            self.dual_system.run_dual_real_time(camera_id)
            
        except KeyboardInterrupt:
            print("🛑 Data collection stopped by user")
        except Exception as e:
            print(f"❌ Error in data collection: {e}")
        finally:
            self.stop_event.set()
    
    def _training_worker(self):
        """Worker thread for periodic model retraining."""
        print("🧠 Training worker started")
        
        while not self.stop_event.is_set():
            try:
                # Wait for retrain interval
                if self.stop_event.wait(self.retrain_interval):
                    break  # Stop event was set
                
                # Check if we should retrain
                if self._should_retrain():
                    self._perform_retraining()
                
            except Exception as e:
                print(f"❌ Error in training worker: {e}")
        
        print("🧠 Training worker stopped")
    
    def _should_retrain(self) -> bool:
        """Check if model should be retrained."""
        if self.is_training:
            return False
        
        # Count new samples collected
        new_samples = self._count_new_samples()
        
        print(f"📊 Training check: {new_samples} new samples")
        
        return new_samples >= self.min_new_samples
    
    def _count_new_samples(self) -> int:
        """Count newly collected samples."""
        # Count images in output directories that could be used for training
        left_data = Path("output/data/left")
        right_data = Path("output/data/right")
        
        count = 0
        for data_dir in [left_data, right_data]:
            if data_dir.exists():
                # Count detection JSON files (each represents a training sample)
                count += len(list(data_dir.glob("detection_*.json")))
        
        return count
    
    def _perform_retraining(self):
        """Perform model retraining with collected data."""
        if self.is_training:
            return
        
        self.is_training = True
        self.training_generation += 1
        
        print("🔥 Starting model retraining...")
        print(f"   Generation: {self.training_generation}")
        print(f"   Time since last training: {time.time() - self.last_retrain_time:.1f}s")
        
        try:
            # 1. Prepare new training data
            self._prepare_live_training_data()
            
            # 2. Create augmented variants
            self._create_augmented_variants()
            
            # 3. Train new model
            new_model_path = self._train_new_model()
            
            # 4. Update current model if training was successful
            if new_model_path and Path(new_model_path).exists():
                self.current_model_path = new_model_path
                print(f"✅ Model updated: {new_model_path}")
                
                # Update dual system with new model
                if self.dual_system:
                    self._update_dual_system_model(new_model_path)
            
            self.last_retrain_time = time.time()
            
        except Exception as e:
            print(f"❌ Retraining failed: {e}")
        finally:
            self.is_training = False
    
    def _prepare_live_training_data(self):
        """Prepare training data from collected live data."""
        print("📋 Preparing live training data...")
        
        # Copy recent detection data to training directory
        source_dirs = [
            Path("output/data/left"),
            Path("output/data/right")
        ]
        
        train_images = self.live_data_dir / "images" / "train"
        train_labels = self.live_data_dir / "labels" / "train"
        
        sample_count = 0
        
        for source_dir in source_dirs:
            if not source_dir.exists():
                continue
            
            # Get recent detection files
            detection_files = sorted(
                source_dir.glob("detection_*.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )[:100]  # Take last 100 samples per side
            
            for det_file in detection_files:
                # Load detection data
                with open(det_file) as f:
                    detection_data = json.load(f)
                
                # Find corresponding frame image
                timestamp = det_file.stem.split('_')[1]
                frame_file = source_dir / f"frame_{timestamp}.jpg"
                
                if frame_file.exists():
                    # Copy image
                    target_image = train_images / f"live_{sample_count:06d}.jpg"
                    shutil.copy2(frame_file, target_image)
                    
                    # Create YOLO label file
                    self._create_yolo_label(detection_data, train_labels / f"live_{sample_count:06d}.txt")
                    
                    sample_count += 1
        
        print(f"📊 Prepared {sample_count} live training samples")
    
    def _create_yolo_label(self, detection_data: Dict, label_path: Path):
        """Create YOLO format label file from detection data."""
        # This is a simplified version - you may need to adapt based on your detection format
        with open(label_path, 'w') as f:
            components = detection_data.get('components', [])
            for comp in components:
                # Convert to YOLO format: class_id center_x center_y width height
                bbox = comp.get('bbox', [0, 0, 100, 100])
                x1, y1, x2, y2 = bbox
                
                # Normalize to image dimensions (assuming 640x640)
                img_w, img_h = 640, 640
                center_x = ((x1 + x2) / 2) / img_w
                center_y = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                
                # Get class ID (you may need to map component types to class IDs)
                class_id = self._get_class_id(comp.get('component_type', 'unknown'))
                
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
    
    def _get_class_id(self, component_type: str) -> int:
        """Map component type to class ID."""
        # This should match your training classes
        class_map = {
            'alarm': 0, 'battery_holder': 1, 'button': 2, 'buzzer': 3,
            'lamp': 4, 'led': 5, 'music_circuit': 6, 'photoresistor': 7,
            'resistor': 8, 'speaker': 9, 'switch': 10, 'wire': 11
        }
        return class_map.get(component_type, 0)
    
    def _create_augmented_variants(self):
        """Create augmented training variants for high camera."""
        print("🔄 Creating augmented variants...")
        
        source_dir = str(self.live_data_dir)
        
        # Create resolution variants
        resolutions = [(416, 416), (832, 832)]
        self.training_prep.create_resolution_variants(source_dir, resolutions)
        
        # Create zoom variants
        zoom_factors = [0.8, 0.6]
        self.training_prep.create_zoomed_out_variants(source_dir, zoom_factors)
    
    def _train_new_model(self) -> Optional[str]:
        """Train new model with live data."""
        print("🚀 Training new model...")
        
        # Create data.yaml for training
        data_yaml = self.live_data_dir / "data.yaml"
        with open(data_yaml, 'w') as f:
            f.write(f"""
train: {self.live_data_dir}/images/train
val: {self.live_data_dir}/images/val
nc: 12
names: ['alarm', 'battery_holder', 'button', 'buzzer', 'lamp', 'led', 
        'music_circuit', 'photoresistor', 'resistor', 'speaker', 'switch', 'wire']
""")
        
        # Train model
        model = YOLO(self.current_model_path)  # Start from current model
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = model.train(
            data=str(data_yaml),
            epochs=20,  # Shorter epochs for live training
            imgsz=640,
            device='cpu',  # Use CPU for live training
            project='live_high_camera_training',
            name=f'live_model_{timestamp}',
            patience=5,
            save=True,
            verbose=True,
            conf=0.15,
            iou=0.4,
            augment=True,
            mosaic=0.8,
            mixup=0.1,
        )
        
        # Return path to best model
        best_model = f"live_high_camera_training/live_model_{timestamp}/weights/best.pt"
        return best_model if Path(best_model).exists() else None
    
    def _update_dual_system_model(self, new_model_path: str):
        """Update the dual camera system with new model."""
        print(f"🔄 Updating dual system model: {new_model_path}")
        
        # This would require modifying the dual system to support hot-swapping models
        # For now, just log the update
        print("⚠️  Model hot-swapping not implemented yet - restart system to use new model")
    
    def stop(self):
        """Stop the live training system."""
        print("🛑 Stopping live high camera trainer...")
        
        self.stop_event.set()
        
        if self.dual_system:
            self.dual_system.cleanup()
        
        if self.data_collection_thread:
            self.data_collection_thread.join(timeout=5)
        
        if self.training_thread:
            self.training_thread.join(timeout=5)
        
        print("✅ Live trainer stopped")


def main():
    parser = argparse.ArgumentParser(description="Live High Camera Training System")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--retrain-interval", type=int, default=600, help="Seconds between retraining (default: 600)")
    parser.add_argument("--min-samples", type=int, default=50, help="Min new samples before retrain (default: 50)")
    parser.add_argument("--collect-only", action="store_true", help="Only collect data, don't retrain")
    
    args = parser.parse_args()
    
    print("🎯 Live High Camera Training System")
    print("=" * 50)
    
    # Initialize trainer
    trainer = LiveHighCameraTrainer(
        retrain_interval=args.retrain_interval,
        min_new_samples=args.min_samples,
        collect_only=args.collect_only
    )
    
    try:
        # Start data collection and training
        trainer.start_data_collection(args.camera)
        
        # Keep main thread alive
        while not trainer.stop_event.is_set():
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    finally:
        trainer.stop()


if __name__ == "__main__":
    main()
