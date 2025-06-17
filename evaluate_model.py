"""
Model evaluation script for Snap Circuit component detection.
Calculates MSE in pixels between predicted and actual bounding boxes.
"""

import os
import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json
from dataclasses import dataclass
import matplotlib.pyplot as plt

from models.component_detector import ComponentDetector
from data_structures import BoundingBox
from config import COMPONENT_CLASSES


@dataclass
class GroundTruthBox:
    """Ground truth bounding box in YOLO format."""
    class_id: int
    center_x: float  # Normalized [0,1]
    center_y: float  # Normalized [0,1] 
    width: float     # Normalized [0,1]
    height: float    # Normalized [0,1]
    
    def to_pixel_bbox(self, img_width: int, img_height: int) -> BoundingBox:
        """Convert normalized YOLO format to pixel BoundingBox."""
        # Convert normalized coordinates to pixels
        center_x_px = self.center_x * img_width
        center_y_px = self.center_y * img_height
        width_px = self.width * img_width
        height_px = self.height * img_height
        
        # Convert center+size to corner coordinates
        x1 = center_x_px - width_px / 2
        y1 = center_y_px - height_px / 2
        x2 = center_x_px + width_px / 2
        y2 = center_y_px + height_px / 2
        
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    mse_pixels: float
    rmse_pixels: float
    mae_pixels: float
    mse_by_class: Dict[str, float]
    rmse_by_class: Dict[str, float]
    mae_by_class: Dict[str, float]
    total_predictions: int
    matched_predictions: int
    precision: float
    recall: float
    f1_score: float


class ModelEvaluator:
    """Evaluates trained YOLO model performance."""
    
    def __init__(self, model_path: str, data_yaml_path: str, conf_threshold: float = 0.25):
        """
        Initialize the model evaluator.
        
        Args:
            model_path: Path to trained model weights
            data_yaml_path: Path to dataset configuration YAML
            conf_threshold: Confidence threshold for predictions
        """
        self.model_path = model_path
        self.data_yaml_path = data_yaml_path
        self.conf_threshold = conf_threshold
        
        # Load dataset configuration
        with open(data_yaml_path, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        self.dataset_root = Path(data_yaml_path).parent
        self.class_names = [self.data_config['names'][i] for i in range(self.data_config['nc'])]
        
        # Initialize detector
        self.detector = ComponentDetector(model_path, conf_threshold=conf_threshold)
        
        print(f"Loaded model: {model_path}")
        print(f"Dataset: {data_yaml_path}")
        print(f"Classes: {self.class_names}")
        print(f"Confidence threshold: {conf_threshold}")
    
    def load_ground_truth(self, label_file: Path) -> List[GroundTruthBox]:
        """Load ground truth boxes from YOLO label file."""
        ground_truth = []
        
        if not label_file.exists():
            return ground_truth
        
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        center_x = float(parts[1])
                        center_y = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        ground_truth.append(GroundTruthBox(
                            class_id=class_id,
                            center_x=center_x,
                            center_y=center_y,
                            width=width,
                            height=height
                        ))
        
        return ground_truth
    
    def calculate_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        # Calculate intersection
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def match_predictions_to_ground_truth(self, 
                                        predictions: List[BoundingBox], 
                                        pred_classes: List[int],
                                        ground_truth: List[GroundTruthBox],
                                        img_width: int, 
                                        img_height: int,
                                        iou_threshold: float = 0.5) -> List[Tuple[BoundingBox, BoundingBox, int]]:
        """
        Match predicted boxes to ground truth boxes based on IoU and class.
        
        Returns:
            List of (predicted_box, ground_truth_box, class_id) tuples for matched pairs
        """
        # Convert ground truth to pixel coordinates
        gt_pixel_boxes = [gt.to_pixel_bbox(img_width, img_height) for gt in ground_truth]
        
        matches = []
        used_gt_indices = set()
        
        for i, (pred_box, pred_class) in enumerate(zip(predictions, pred_classes)):
            best_iou = 0.0
            best_gt_idx = -1
            
            for j, (gt_box, gt) in enumerate(zip(gt_pixel_boxes, ground_truth)):
                # Skip if already matched or wrong class
                if j in used_gt_indices or pred_class != gt.class_id:
                    continue
                
                iou = self.calculate_iou(pred_box, gt_box)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_gt_idx >= 0:
                matches.append((pred_box, gt_pixel_boxes[best_gt_idx], pred_class))
                used_gt_indices.add(best_gt_idx)
        
        return matches
    
    def calculate_bbox_mse(self, pred_box: BoundingBox, gt_box: BoundingBox) -> float:
        """Calculate MSE between two bounding boxes (in pixels)."""
        # Calculate squared differences for each coordinate
        mse = ((pred_box.x1 - gt_box.x1) ** 2 + 
               (pred_box.y1 - gt_box.y1) ** 2 +
               (pred_box.x2 - gt_box.x2) ** 2 + 
               (pred_box.y2 - gt_box.y2) ** 2) / 4
        
        return mse
    
    def calculate_bbox_mae(self, pred_box: BoundingBox, gt_box: BoundingBox) -> float:
        """Calculate MAE between two bounding boxes (in pixels)."""
        # Calculate absolute differences for each coordinate
        mae = (abs(pred_box.x1 - gt_box.x1) + 
               abs(pred_box.y1 - gt_box.y1) +
               abs(pred_box.x2 - gt_box.x2) + 
               abs(pred_box.y2 - gt_box.y2)) / 4
        
        return mae
    
    def evaluate_dataset(self, split: str = 'val') -> EvaluationMetrics:
        """
        Evaluate model on a dataset split.
        
        Args:
            split: Dataset split to evaluate ('train', 'val', or 'test')
            
        Returns:
            EvaluationMetrics containing all evaluation results
        """
        print(f"Evaluating on {split} split...")
        
        # Get image and label directories
        images_dir = self.dataset_root / f"images/{split}"
        labels_dir = self.dataset_root / f"labels/{split}"
        
        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")
        if not labels_dir.exists():
            raise ValueError(f"Labels directory not found: {labels_dir}")
        
        # Collect all evaluation data
        all_mse_values = []
        all_mae_values = []
        mse_by_class = {class_name: [] for class_name in self.class_names}
        mae_by_class = {class_name: [] for class_name in self.class_names}
        
        total_predictions = 0
        matched_predictions = 0
        total_ground_truth = 0
        
        # Process each image
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        for i, image_file in enumerate(image_files):
            if i % 10 == 0:
                print(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
            
            # Load image
            image = cv2.imread(str(image_file))
            if image is None:
                print(f"Warning: Could not load image {image_file}")
                continue
            
            img_height, img_width = image.shape[:2]
            
            # Load ground truth
            label_file = labels_dir / f"{image_file.stem}.txt"
            ground_truth = self.load_ground_truth(label_file)
            total_ground_truth += len(ground_truth)
            
            if not ground_truth:
                continue
            
            # Run detection
            detections = self.detector.detect(image)
            total_predictions += len(detections)
            
            if not detections:
                continue
            
            # Extract predictions
            pred_boxes = [det.bbox for det in detections]
            pred_classes = [self.class_names.index(det.label) if det.label in self.class_names else -1 
                          for det in detections]
            
            # Match predictions to ground truth
            matches = self.match_predictions_to_ground_truth(
                pred_boxes, pred_classes, ground_truth, img_width, img_height
            )
            matched_predictions += len(matches)
            
            # Calculate metrics for matched pairs
            for pred_box, gt_box, class_id in matches:
                mse = self.calculate_bbox_mse(pred_box, gt_box)
                mae = self.calculate_bbox_mae(pred_box, gt_box)
                
                all_mse_values.append(mse)
                all_mae_values.append(mae)
                
                if class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                    mse_by_class[class_name].append(mse)
                    mae_by_class[class_name].append(mae)
        
        # Calculate overall metrics
        if all_mse_values:
            overall_mse = np.mean(all_mse_values)
            overall_rmse = np.sqrt(overall_mse)
            overall_mae = np.mean(all_mae_values)
        else:
            overall_mse = overall_rmse = overall_mae = float('inf')
        
        # Calculate per-class metrics
        mse_by_class_avg = {}
        rmse_by_class_avg = {}
        mae_by_class_avg = {}
        
        for class_name in self.class_names:
            if mse_by_class[class_name]:
                mse_by_class_avg[class_name] = np.mean(mse_by_class[class_name])
                rmse_by_class_avg[class_name] = np.sqrt(mse_by_class_avg[class_name])
                mae_by_class_avg[class_name] = np.mean(mae_by_class[class_name])
            else:
                mse_by_class_avg[class_name] = float('inf')
                rmse_by_class_avg[class_name] = float('inf')
                mae_by_class_avg[class_name] = float('inf')
        
        # Calculate detection metrics
        precision = matched_predictions / total_predictions if total_predictions > 0 else 0.0
        recall = matched_predictions / total_ground_truth if total_ground_truth > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"\nEvaluation completed!")
        print(f"Total images: {len(image_files)}")
        print(f"Total predictions: {total_predictions}")
        print(f"Total ground truth: {total_ground_truth}")
        print(f"Matched predictions: {matched_predictions}")
        
        return EvaluationMetrics(
            mse_pixels=overall_mse,
            rmse_pixels=overall_rmse,
            mae_pixels=overall_mae,
            mse_by_class=mse_by_class_avg,
            rmse_by_class=rmse_by_class_avg,
            mae_by_class=mae_by_class_avg,
            total_predictions=total_predictions,
            matched_predictions=matched_predictions,
            precision=precision,
            recall=recall,
            f1_score=f1_score
        )
    
    def print_results(self, metrics: EvaluationMetrics):
        """Print evaluation results in a formatted way."""
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        print(f"\n📊 OVERALL BOUNDING BOX ACCURACY:")
        print(f"   MSE (pixels²):  {metrics.mse_pixels:.2f}")
        print(f"   RMSE (pixels):  {metrics.rmse_pixels:.2f}")
        print(f"   MAE (pixels):   {metrics.mae_pixels:.2f}")
        
        print(f"\n🎯 DETECTION PERFORMANCE:")
        print(f"   Precision:      {metrics.precision:.3f}")
        print(f"   Recall:         {metrics.recall:.3f}")
        print(f"   F1-Score:       {metrics.f1_score:.3f}")
        print(f"   Matched/Total:  {metrics.matched_predictions}/{metrics.total_predictions}")
        
        print(f"\n📋 PER-CLASS BOUNDING BOX ACCURACY:")
        for class_name in self.class_names:
            mse = metrics.mse_by_class[class_name]
            rmse = metrics.rmse_by_class[class_name]
            mae = metrics.mae_by_class[class_name]
            
            if mse != float('inf'):
                print(f"   {class_name:15} - RMSE: {rmse:6.2f}px, MAE: {mae:6.2f}px")
            else:
                print(f"   {class_name:15} - No matched detections")
        
        print("="*60)
    
    def save_results(self, metrics: EvaluationMetrics, output_file: str):
        """Save evaluation results to JSON file."""
        results = {
            'overall_metrics': {
                'mse_pixels': float(metrics.mse_pixels),
                'rmse_pixels': float(metrics.rmse_pixels),
                'mae_pixels': float(metrics.mae_pixels),
                'precision': float(metrics.precision),
                'recall': float(metrics.recall),
                'f1_score': float(metrics.f1_score)
            },
            'detection_counts': {
                'total_predictions': metrics.total_predictions,
                'matched_predictions': metrics.matched_predictions
            },
            'per_class_metrics': {
                'mse': {k: float(v) for k, v in metrics.mse_by_class.items()},
                'rmse': {k: float(v) for k, v in metrics.rmse_by_class.items()},
                'mae': {k: float(v) for k, v in metrics.mae_by_class.items()}
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    
    def create_visualization(self, metrics: EvaluationMetrics, output_dir: str):
        """Create visualization plots of the evaluation results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: RMSE by class
        plt.figure(figsize=(12, 8))
        
        classes = []
        rmse_values = []
        
        for class_name, rmse in metrics.rmse_by_class.items():
            if rmse != float('inf'):
                classes.append(class_name)
                rmse_values.append(rmse)
        
        if classes:
            plt.subplot(2, 1, 1)
            bars = plt.bar(classes, rmse_values, color='skyblue', edgecolor='navy', alpha=0.7)
            plt.title('Root Mean Square Error (RMSE) by Component Class', fontsize=14, fontweight='bold')
            plt.xlabel('Component Class', fontsize=12)
            plt.ylabel('RMSE (pixels)', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, rmse_values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
            
            # Plot 2: Detection performance
            plt.subplot(2, 1, 2)
            performance_metrics = ['Precision', 'Recall', 'F1-Score']
            performance_values = [metrics.precision, metrics.recall, metrics.f1_score]
            
            bars = plt.bar(performance_metrics, performance_values, 
                          color=['lightcoral', 'lightgreen', 'lightsalmon'])
            plt.title('Detection Performance Metrics', fontsize=14, fontweight='bold')
            plt.ylabel('Score', fontsize=12)
            plt.ylim(0, 1)
            plt.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, performance_values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved to: {output_dir / 'evaluation_results.png'}")


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Snap Circuit component detection model')
    parser.add_argument('--model', type=str, 
                       default='snap_circuit_training/expanded_snap_circuit_model/weights/best.pt',
                       help='Path to trained model weights')
    parser.add_argument('--data', type=str,
                       default='data/augmented_training/data.yaml',
                       help='Path to dataset configuration YAML')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--output', type=str, default='output/evaluation_results.json',
                       help='Output file for results')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization plots')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Check if data config exists
    if not os.path.exists(args.data):
        print(f"Error: Data config file not found: {args.data}")
        return
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(args.model, args.data)
        
        # Run evaluation
        metrics = evaluator.evaluate_dataset(args.split)
        
        # Print results
        evaluator.print_results(metrics)
        
        # Save results
        evaluator.save_results(metrics, args.output)
        
        # Create visualization if requested
        if args.visualize:
            output_dir = Path(args.output).parent
            evaluator.create_visualization(metrics, output_dir)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 