#!/usr/bin/env python3
"""
Train Clean Model

Train a model using the clean dataset with minimal spatial augmentation
to preserve the perfect spatial balance.

This training approach:
1. Uses the clean balanced dataset
2. Disables problematic spatial augmentations  
3. Preserves left/right balance during training
4. Should result in symmetric detection
"""

import os
from pathlib import Path
from ultralytics import YOLO
import yaml


def train_clean_model():
    """Train model with clean balanced dataset and minimal augmentation"""
    
    print("🧹 CLEAN MODEL TRAINING")
    print("=" * 50)
    print("🎯 Goal: Train with preserved spatial balance")
    print("🚫 Strategy: Disable spatial augmentations that corrupt balance")
    
    # Check if clean dataset exists
    clean_dataset = Path("clean_dual_board_dataset")
    data_yaml = clean_dataset / "data.yaml"
    
    if not data_yaml.exists():
        print(f"❌ Clean dataset not found!")
        print(f"   Expected: {data_yaml}")
        print(f"   Run: python create_clean_training_dataset.py")
        return
    
    print(f"✅ Clean dataset found: {clean_dataset}")
    
    # Verify dataset structure
    train_images = clean_dataset / "images" / "train" 
    val_images = clean_dataset / "images" / "val"
    train_labels = clean_dataset / "labels" / "train"
    val_labels = clean_dataset / "labels" / "val"
    
    train_count = len(list(train_images.glob("*.jpg")))
    val_count = len(list(val_images.glob("*.jpg")))
    
    print(f"📊 Training images: {train_count}")
    print(f"📊 Validation images: {val_count}")
    
    if train_count == 0:
        print(f"❌ No training images found!")
        return
    
    # Initialize model
    print(f"\n🤖 Initializing YOLOv8x model...")
    model = YOLO("yolov8x.pt")
    
    # Training parameters optimized for spatial balance preservation
    train_params = {
        'data': str(data_yaml),
        'epochs': 100,
        'batch': 8,
        'imgsz': 640,
        'patience': 20,
        'save_period': 10,
        'name': 'clean_dual_board_model',
        'project': 'dual_board_training',
        
        # Critical: Disable spatial augmentations that corrupt balance
        'fliplr': 0.0,      # ❌ NO horizontal flipping (corrupts left/right balance)
        'flipud': 0.0,      # ❌ NO vertical flipping
        'mosaic': 0.0,      # ❌ NO mosaic (mixes spatial relationships)  
        'mixup': 0.0,       # ❌ NO mixup (mixes spatial relationships)
        'copy_paste': 0.0,  # ❌ NO copy-paste (moves components around)
        
        # Safe augmentations (preserve spatial relationships)
        'hsv_h': 0.015,     # ✅ Color/hue changes OK
        'hsv_s': 0.3,       # ✅ Saturation changes OK  
        'hsv_v': 0.2,       # ✅ Brightness changes OK
        'degrees': 0,       # ❌ NO rotation (can affect left/right)
        'translate': 0.05,  # ✅ Minimal translation OK
        'scale': 0.1,       # ✅ Minimal scaling OK
        'shear': 0,         # ❌ NO shearing (distorts spatial relationships)
        'perspective': 0.0, # ❌ NO perspective transform
        
        # Optimizer settings
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'cos_lr': True,
        
        # Other settings
        'verbose': True,
        'plots': True,
        'save': True,
        'val': True,
        'amp': True,
        'device': 'cpu',  # Use CPU to ensure consistency
    }
    
    print(f"\n🎯 TRAINING CONFIGURATION:")
    print(f"   📊 Dataset: Clean balanced annotations")
    print(f"   🚫 Spatial augmentations: DISABLED")
    print(f"   ✅ Color augmentations: Minimal")
    print(f"   📈 Epochs: {train_params['epochs']}")
    print(f"   🎯 Goal: Preserve left/right spatial balance")
    
    print(f"\n🚀 Starting clean model training...")
    print(f"   Expected time: 30-60 minutes")
    print(f"   This should solve the asymmetric detection!")
    
    try:
        # Start training
        results = model.train(**train_params)
        
        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"   📁 Model saved to: dual_board_training/clean_dual_board_model/weights/best.pt")
        print(f"   📊 Training results: {results}")
        
        # Copy best model to convenient location
        best_model = Path("dual_board_training/clean_dual_board_model/weights/best.pt")
        if best_model.exists():
            clean_model_dest = Path("models/weights/clean_dual_board_model.pt")
            clean_model_dest.parent.mkdir(exist_ok=True)
            
            import shutil
            shutil.copy2(best_model, clean_model_dest)
            print(f"   ✅ Model copied to: {clean_model_dest}")
            
            print(f"\n💡 NEXT STEPS:")
            print(f"   🧪 Test the clean model:")
            print(f"   python test_clean_model.py")
            print(f"   ")
            print(f"   🎯 Expected results:")
            print(f"   • LEFT battery holder detection: ✅")
            print(f"   • RIGHT battery holder detection: ✅") 
            print(f"   • Symmetric wire detection: ✅")
            
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False


if __name__ == "__main__":
    train_clean_model()
