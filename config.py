"""
Configuration settings for the Snap Circuit computer vision system.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models" / "weights"
DATA_DIR = PROJECT_ROOT / "data"
TRAINING_DATA_DIR = DATA_DIR / "training"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Create directories if they don't exist
for dir_path in [MODELS_DIR, DATA_DIR, TRAINING_DATA_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# YOLOv8 Model Configuration
YOLO_CONFIG = {
    "model_path": MODELS_DIR / "snap_circuit_yolov8.pt",
    "pretrained_model": "yolov8x.pt",
    "confidence_threshold": 0.1,  # Lowered for better detection
    "iou_threshold": 0.45,
    "image_size": 640,
    "device": "cuda"  # GPU acceleration with RTX 4070 Ti
}

# Training Configuration
TRAINING_CONFIG = {
    "epochs": 100,
    "batch_size": 16,
    "learning_rate": 0.01,
    "patience": 50,
    "data_yaml": TRAINING_DATA_DIR / "data.yaml"
}

# Component Ontology - All detectable Snap Circuit components
COMPONENT_CLASSES = [
    "wire",
    "switch", 
    "button",
    "battery_holder",
    "led",
    "speaker",
    "music_circuit", 
    "motor",
    "resistor",
    "connection_node",
    "lamp",
    "fan",
    "buzzer",
    "photoresistor",
    "microphone",
    "alarm"
]

# Component orientations (in degrees)
ORIENTATIONS = {
    "NORTH": 0,
    "EAST": 90, 
    "SOUTH": 180,
    "WEST": 270
}

# Connection Detection Parameters
CONNECTION_CONFIG = {
    "wire_color_range": {
        "lower": [0, 0, 150],     # Lower HSV range for metallic connections
        "upper": [180, 30, 255]   # Upper HSV range for metallic connections
    },
    "metallic_color_range": {
        "lower": [0, 0, 180],     # Silver/gold metallic snap points
        "upper": [30, 30, 255]
    },
    "min_contour_area": 20,       # Smaller for fine connection points
    "max_contour_area": 2000,     # Reduced for Snap Circuit scale
    "line_thickness": 2,
    "connection_proximity_threshold": 40,  # Increased for Snap Circuit spacing
    "skeleton_kernel_size": 3,
    "snap_point_detection": True,  # Enable detection of hexagonal snap points
    "component_margin": 15        # Margin around components for connection detection
}

# Circuit Analysis Parameters
CIRCUIT_CONFIG = {
    "power_components": ["battery_holder"],
    "output_components": ["led", "speaker", "motor", "lamp", "fan", "buzzer"],
    "input_components": ["button", "switch", "photoresistor", "microphone"],
    "passive_components": ["resistor", "wire", "connection_node"],
    "min_circuit_length": 2  # Minimum components for valid circuit
}

# Video Processing
VIDEO_CONFIG = {
    "camera_id": 0,
    "fps": 30,
    "resolution": (1920, 1080),
    "buffer_size": 1,  # Minimize latency
    "processing_interval": 1.0,  # Process every 1 second (adjustable 1-2 seconds)
    "skip_frames": True  # Skip frames between processing intervals
}

# Output Format Settings
OUTPUT_CONFIG = {
    "save_annotated_frames": True,
    "save_detection_data": True,
    "output_format": "json",  # json or yaml
    "real_time_display": True
}

# Coordinate system for robot integration
ROBOT_CONFIG = {
    "camera_height": 500,  # mm above the board
    "board_origin": (0, 0),  # Camera coordinate origin
    "pixel_to_mm_ratio": 0.5,  # Calibration required
    "coordinate_system": "top_down"
} 