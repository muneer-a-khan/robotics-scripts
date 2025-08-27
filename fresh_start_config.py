"""
Fresh Start Configuration for Snap Circuit Component Detection
Complete restart of the training and detection pipeline
"""

import os
from pathlib import Path

class FreshStartConfig:
    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_ROOT = PROJECT_ROOT / "data"
    
    # Dataset paths
    TRAINING_DATA = DATA_ROOT / "augmented_training"
    INDIVIDUAL_COMPONENTS = DATA_ROOT / "individual_components"
    
    # Output paths
    OUTPUT_DIR = PROJECT_ROOT / "output"
    MODELS_DIR = OUTPUT_DIR / "models"
    RESULTS_DIR = OUTPUT_DIR / "results"
    VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
    
    # Training configuration
    MODEL_SIZE = "n"  # n, s, m, l, x
    EPOCHS = 100
    BATCH_SIZE = 16  # Back to original for GPU training
    IMAGE_SIZE = 640
    PATIENCE = 20  # Early stopping patience
    
    # Detection configuration
    CONFIDENCE_THRESHOLD = 0.25
    NMS_THRESHOLD = 0.45
    MAX_DETECTIONS = 100
    
    # Component classes (16 total)
    COMPONENT_CLASSES = [
        "wire", "switch", "button", "battery_holder", "led", 
        "speaker", "music_circuit", "motor", "resistor", 
        "connection_node", "lamp", "fan", "buzzer", 
        "photoresistor", "microphone", "alarm"
    ]
    
    # Connection detection settings
    CONNECTION_DETECTION_ENABLED = True
    CONNECTION_DISTANCE_THRESHOLD = 50  # pixels
    CONNECTION_ANGLE_THRESHOLD = 30  # degrees
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.OUTPUT_DIR,
            cls.MODELS_DIR,
            cls.RESULTS_DIR,
            cls.VISUALIZATIONS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")
    
    @classmethod
    def get_data_yaml_path(cls):
        """Get the path to the data.yaml file"""
        return cls.TRAINING_DATA / "data.yaml"
    
    @classmethod
    def get_model_save_path(cls, model_name="snap_circuit_detector"):
        """Get the path to save the trained model"""
        return cls.MODELS_DIR / f"{model_name}.pt"
    
    @classmethod
    def get_results_path(cls, experiment_name):
        """Get the path for experiment results"""
        return cls.RESULTS_DIR / experiment_name

# Initialize directories
FreshStartConfig.create_directories() 