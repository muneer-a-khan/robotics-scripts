#!/usr/bin/env python3
"""
Simple script to train YOLOv8 on Snap Circuit data.
"""

from models.component_detector import ComponentDetector
import os

def train_snap_circuit_model():
    print("🚀 Starting Snap Circuit Model Training...")
    
    # Check if dataset exists
    data_yaml = "data/training/data.yaml"
    if not os.path.exists(data_yaml):
        print("❌ Dataset not found. Run create_training_data.py first!")
        return
    
    # Initialize detector
    detector = ComponentDetector()
    
    # Train the model
    print(f"Training on dataset: {data_yaml}")
    results = detector.train(
        data_yaml_path=data_yaml,
        epochs=50,        # Start with fewer epochs for testing
        imgsz=640,
        batch=1,          # Small batch size for single image
        lr0=0.01,
        patience=20,
        device='cpu'      # Force CPU training
    )
    
    print("✅ Training complete!")
    print("Model saved to: models/weights/snap_circuit_yolov8.pt")
    
    return results

if __name__ == "__main__":
    train_snap_circuit_model()
