"""
YOLOv8-based component detector for Snap Circuit pieces.
"""

import os
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2
from ultralytics import YOLO
import torch

from config import YOLO_CONFIG, COMPONENT_CLASSES, TRAINING_CONFIG
from data_structures import (
    ComponentDetection, BoundingBox, ComponentType, 
    bbox_from_yolo, create_component_id, calculate_orientation
)


class ComponentDetector:
    """
    YOLOv8-based detector for Snap Circuit components.
    """
    
    def __init__(self, model_path: Optional[str] = None, conf_threshold: Optional[float] = None):
        """
        Initialize the component detector.
        
        Args:
            model_path: Path to trained YOLOv8 model. If None, uses config default.
            conf_threshold: Confidence threshold override. If None, uses config default.
        """
        self.model_path = model_path or YOLO_CONFIG["model_path"]
        self.confidence_threshold = conf_threshold or YOLO_CONFIG["confidence_threshold"]
        self.iou_threshold = YOLO_CONFIG["iou_threshold"]
        self.image_size = YOLO_CONFIG["image_size"]
        self.device = YOLO_CONFIG["device"]
        
        # Component class mapping
        self.class_names = COMPONENT_CLASSES
        self.class_to_type = {name: ComponentType(name) for name in COMPONENT_CLASSES}
        
        # Initialize model
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load the YOLOv8 model."""
        try:
            # Check if custom trained model exists
            if os.path.exists(self.model_path):
                print(f"Loading trained model from {self.model_path}")
                self.model = YOLO(self.model_path)
            else:
                print(f"Trained model not found at {self.model_path}")
                print(f"Loading pretrained model: {YOLO_CONFIG['pretrained_model']}")
                self.model = YOLO(YOLO_CONFIG["pretrained_model"])
                
                # Create placeholder for custom training
                print("Note: You'll need to train the model with your Snap Circuit data")
                print("Use the training/data_preparation.py script to prepare your dataset")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to YOLOv8n...")
            self.model = YOLO("yolov8n.pt")
    
    def detect(self, image: np.ndarray) -> List[ComponentDetection]:
        """
        Detect components in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detected components
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
            
        # Run inference
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False
        )
        
        # Parse results
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            # Get detection data
            xyxy = boxes.xyxy.cpu().numpy()  # Bounding boxes
            confidences = boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = boxes.cls.cpu().numpy().astype(int)  # Class IDs
            
            # Convert to ComponentDetection objects
            for i, (box, conf, class_id) in enumerate(zip(xyxy, confidences, class_ids)):
                # Create bounding box
                bbox = BoundingBox(
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3])
                )
                
                # Get component type and label
                if class_id < len(self.class_names):
                    label = self.class_names[class_id]
                    component_type = self.class_to_type[label]
                else:
                    label = f"unknown_{class_id}"
                    component_type = ComponentType.WIRE  # Default fallback
                
                # Calculate orientation (basic heuristic)
                orientation = calculate_orientation(bbox)
                
                # Create component detection
                detection = ComponentDetection(
                    id=create_component_id(component_type, i),
                    label=label,
                    bbox=bbox,
                    orientation=orientation,
                    confidence=float(conf),
                    component_type=component_type,
                    metadata={
                        "yolo_class_id": int(class_id),
                        "detection_index": i
                    }
                )
                
                detections.append(detection)
        
        return detections
    
    def train(self, data_yaml_path: str, **kwargs) -> None:
        """
        Train the YOLOv8 model on custom data.
        
        Args:
            data_yaml_path: Path to YOLO dataset configuration file
            **kwargs: Additional training parameters
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Set default training parameters
        train_params = {
            "data": data_yaml_path,
            "epochs": TRAINING_CONFIG["epochs"],
            "imgsz": self.image_size,
            "batch": TRAINING_CONFIG["batch_size"],
            "lr0": TRAINING_CONFIG["learning_rate"],
            "patience": TRAINING_CONFIG["patience"],
            "device": self.device,
            "project": "snap_circuit_training",
            "name": "yolov8_snap_circuit"
        }
        
        # Override with any provided kwargs
        train_params.update(kwargs)
        
        print(f"Starting training with parameters: {train_params}")
        
        # Train the model
        results = self.model.train(**train_params)
        
        # Save the trained model
        best_model_path = results.save_dir / "weights" / "best.pt"
        if best_model_path.exists():
            # Copy to our models directory
            import shutil
            shutil.copy(best_model_path, self.model_path)
            print(f"Trained model saved to {self.model_path}")
        
        return results
    
    def validate(self, data_yaml_path: str) -> Dict[str, Any]:
        """
        Validate the model on test data.
        
        Args:
            data_yaml_path: Path to YOLO dataset configuration file
            
        Returns:
            Validation metrics
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
            
        results = self.model.val(
            data=data_yaml_path,
            imgsz=self.image_size,
            device=self.device
        )
        
        return results.results_dict if hasattr(results, 'results_dict') else {}
    
    def export(self, format: str = "onnx") -> str:
        """
        Export the model to different formats.
        
        Args:
            format: Export format (onnx, torchscript, etc.)
            
        Returns:
            Path to exported model
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
            
        export_path = self.model.export(format=format)
        print(f"Model exported to: {export_path}")
        return export_path
    
    def benchmark(self, image: np.ndarray, num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark detection performance.
        
        Args:
            image: Test image
            num_runs: Number of inference runs
            
        Returns:
            Performance metrics
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Warmup
        for _ in range(10):
            self.detect(image)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            detections = self.detect(image)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        fps = 1.0 / avg_time
        
        return {
            "avg_inference_time": avg_time,
            "fps": fps,
            "total_time": total_time,
            "num_runs": num_runs,
            "num_detections": len(detections) if 'detections' in locals() else 0
        }
    
    def annotate_image(self, image: np.ndarray, detections: List[ComponentDetection]) -> np.ndarray:
        """
        Draw detection results on image.
        
        Args:
            image: Input image
            detections: List of detections to draw
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        for detection in detections:
            bbox = detection.bbox
            
            # Draw bounding box
            cv2.rectangle(
                annotated,
                (int(bbox.x1), int(bbox.y1)),
                (int(bbox.x2), int(bbox.y2)),
                (0, 255, 0),  # Green
                2
            )
            
            # Draw label and confidence
            label_text = f"{detection.label}: {detection.confidence:.2f}"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background rectangle for text
            cv2.rectangle(
                annotated,
                (int(bbox.x1), int(bbox.y1) - label_size[1] - 10),
                (int(bbox.x1) + label_size[0], int(bbox.y1)),
                (0, 255, 0),
                -1
            )
            
            # Text
            cv2.putText(
                annotated,
                label_text,
                (int(bbox.x1), int(bbox.y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # Black text
                2
            )
            
            # Draw center point
            center = bbox.center
            cv2.circle(
                annotated,
                (int(center[0]), int(center[1])),
                3,
                (255, 0, 0),  # Blue
                -1
            )
        
        return annotated


def test_detector():
    """Test function for the component detector."""
    detector = ComponentDetector()
    
    # Create a test image (you would use real camera feed)
    test_image = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.putText(test_image, "Test Image", (50, 320), 
               cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    
    # Run detection
    detections = detector.detect(test_image)
    print(f"Detected {len(detections)} components")
    
    # Annotate and display
    annotated = detector.annotate_image(test_image, detections)
    cv2.imshow("Test Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_detector() 