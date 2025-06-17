#!/usr/bin/env python3
"""
Test the precision-focused model on examples with visual output and detailed metrics.
"""

import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Dict, Any
import yaml

from models.component_detector import ComponentDetector
from data_structures import BoundingBox, ComponentDetection
from evaluate_model import ModelEvaluator, GroundTruthBox


class PrecisionModelTester:
    """Test the precision-focused model with visual output."""
    
    def __init__(self, model_path: str, data_yaml_path: str, conf_threshold: float = 0.2):
        """Initialize the tester with the trained model."""
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
        
        print(f"🔧 Model: {model_path}")
        print(f"📊 Confidence Threshold: {conf_threshold}")
        print(f"📋 Classes: {self.class_names}")
    
    def load_ground_truth_boxes(self, label_file: Path) -> List[GroundTruthBox]:
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
        """Calculate IoU between two bounding boxes."""
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
    
    def create_visual_prediction(self, image_path: str, output_dir: str = "output/test_predictions"):
        """Create visual prediction with ground truth comparison."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        img_height, img_width = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get predictions
        detections = self.detector.detect(image)
        
        # Load ground truth
        image_name = Path(image_path).stem
        # Get the split (val/train/test) from the image path
        split_name = Path(image_path).parent.name
        label_file = self.dataset_root / "labels" / split_name / f"{image_name}.txt"
        ground_truth = self.load_ground_truth_boxes(label_file)
        
        # Convert ground truth to pixel coordinates
        gt_pixel_boxes = []
        for gt in ground_truth:
            gt_box = gt.to_pixel_bbox(img_width, img_height)
            gt_pixel_boxes.append((gt_box, gt.class_id))
        
        # Calculate metrics for this image
        metrics = self.calculate_image_metrics(detections, ground_truth, img_width, img_height)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Left: Predictions
        ax1.imshow(image_rgb)
        ax1.set_title(f"Predictions (Conf ≥ {self.conf_threshold})\n"
                     f"Found: {len(detections)} objects", fontsize=14, fontweight='bold')
        
        # Draw predicted boxes
        for detection in detections:
            bbox = detection.bbox
            rect = patches.Rectangle(
                (bbox.x1, bbox.y1), 
                bbox.x2 - bbox.x1, 
                bbox.y2 - bbox.y1,
                linewidth=3, 
                edgecolor='red', 
                facecolor='none'
            )
            ax1.add_patch(rect)
            
            # Add label with confidence
            ax1.text(
                bbox.x1, bbox.y1 - 5,
                f"{detection.label} ({detection.confidence:.2f})",
                fontsize=12, color='red', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
            )
        
        # Right: Ground Truth
        ax2.imshow(image_rgb)
        ax2.set_title(f"Ground Truth\n"
                     f"Actual: {len(ground_truth)} objects", fontsize=14, fontweight='bold')
        
        # Draw ground truth boxes
        for gt_box, class_id in gt_pixel_boxes:
            rect = patches.Rectangle(
                (gt_box.x1, gt_box.y1),
                gt_box.x2 - gt_box.x1,
                gt_box.y2 - gt_box.y1,
                linewidth=3,
                edgecolor='green',
                facecolor='none'
            )
            ax2.add_patch(rect)
            
            # Add ground truth label
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            ax2.text(
                gt_box.x1, gt_box.y1 - 5,
                f"{class_name}",
                fontsize=12, color='green', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
            )
        
        # Remove axes
        ax1.axis('off')
        ax2.axis('off')
        
        # Add metrics text
        metrics_text = (
            f"📊 Image Metrics:\n"
            f"Precision: {metrics['precision']:.3f}\n"
            f"Recall: {metrics['recall']:.3f}\n"
            f"F1-Score: {metrics['f1']:.3f}\n"
            f"IoU (avg): {metrics['avg_iou']:.3f}\n"
            f"Matched: {metrics['matched']}/{metrics['total_pred']}\n"
            f"RMSE: {metrics['rmse']:.1f}px"
        )
        
        plt.figtext(0.02, 0.02, metrics_text, fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        # Save visualization
        os.makedirs(output_dir, exist_ok=True)
        output_file = Path(output_dir) / f"{image_name}_prediction.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📁 Saved: {output_file}")
        return metrics, output_file
    
    def calculate_image_metrics(self, detections: List[ComponentDetection], 
                               ground_truth: List[GroundTruthBox], 
                               img_width: int, img_height: int) -> Dict[str, float]:
        """Calculate metrics for a single image."""
        # Convert ground truth to pixel coordinates
        gt_pixel_boxes = [gt.to_pixel_bbox(img_width, img_height) for gt in ground_truth]
        
        # Match predictions to ground truth
        matches = []
        used_gt_indices = set()
        total_iou = 0.0
        total_rmse = 0.0
        
        for detection in detections:
            best_iou = 0.0
            best_gt_idx = -1
            best_gt_box = None
            
            for j, (gt_box, gt) in enumerate(zip(gt_pixel_boxes, ground_truth)):
                if j in used_gt_indices:
                    continue
                
                # Check class match
                pred_class_name = detection.label
                gt_class_name = self.class_names[gt.class_id] if gt.class_id < len(self.class_names) else f"class_{gt.class_id}"
                
                if pred_class_name != gt_class_name:
                    continue
                
                iou = self.calculate_iou(detection.bbox, gt_box)
                if iou > best_iou and iou >= 0.5:  # IoU threshold
                    best_iou = iou
                    best_gt_idx = j
                    best_gt_box = gt_box
            
            if best_gt_idx >= 0:
                matches.append((detection.bbox, best_gt_box))
                used_gt_indices.add(best_gt_idx)
                total_iou += best_iou
                
                # Calculate RMSE for this match
                rmse = np.sqrt(((detection.bbox.x1 - best_gt_box.x1) ** 2 + 
                               (detection.bbox.y1 - best_gt_box.y1) ** 2 +
                               (detection.bbox.x2 - best_gt_box.x2) ** 2 + 
                               (detection.bbox.y2 - best_gt_box.y2) ** 2) / 4)
                total_rmse += rmse
        
        # Calculate metrics
        num_matches = len(matches)
        num_predictions = len(detections)
        num_ground_truth = len(ground_truth)
        
        precision = num_matches / num_predictions if num_predictions > 0 else 0.0
        recall = num_matches / num_ground_truth if num_ground_truth > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        avg_iou = total_iou / num_matches if num_matches > 0 else 0.0
        avg_rmse = total_rmse / num_matches if num_matches > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_iou': avg_iou,
            'rmse': avg_rmse,
            'matched': num_matches,
            'total_pred': num_predictions,
            'total_gt': num_ground_truth
        }
    
    def test_on_dataset_split(self, split: str = 'val', max_images: int = 5):
        """Test on multiple images from dataset split."""
        print(f"🧪 Testing on {split} split (max {max_images} images)")
        
        # Get image paths
        if split == 'val':
            split_key = 'val'
        elif split == 'test':
            split_key = 'test'
        else:
            split_key = 'train'
        
        split_path = self.dataset_root / self.data_config[split_key]
        image_dir = split_path
        
        if not image_dir.exists():
            print(f"❌ Image directory not found: {image_dir}")
            return
        
        # Get image files
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        if not image_files:
            print(f"❌ No images found in {image_dir}")
            return
        
        # Limit number of images
        image_files = image_files[:max_images]
        
        print(f"📊 Testing on {len(image_files)} images...")
        
        all_metrics = []
        output_files = []
        
        for image_file in image_files:
            print(f"\n🔍 Processing: {image_file.name}")
            
            try:
                metrics, output_file = self.create_visual_prediction(str(image_file))
                if metrics:
                    all_metrics.append(metrics)
                    output_files.append(output_file)
                    
                    # Print metrics for this image
                    print(f"   Precision: {metrics['precision']:.3f}")
                    print(f"   Recall: {metrics['recall']:.3f}")
                    print(f"   F1-Score: {metrics['f1']:.3f}")
                    print(f"   RMSE: {metrics['rmse']:.1f}px")
                    print(f"   Matches: {metrics['matched']}/{metrics['total_pred']}")
                    
            except Exception as e:
                print(f"   ❌ Error processing {image_file.name}: {e}")
        
        # Calculate overall metrics
        if all_metrics:
            self.print_overall_metrics(all_metrics)
            self.create_summary_visualization(all_metrics, output_files)
        
        return all_metrics, output_files
    
    def print_overall_metrics(self, all_metrics: List[Dict[str, float]]):
        """Print aggregated metrics across all test images."""
        if not all_metrics:
            return
        
        # Aggregate metrics
        total_matched = sum(m['matched'] for m in all_metrics)
        total_predictions = sum(m['total_pred'] for m in all_metrics)
        total_ground_truth = sum(m['total_gt'] for m in all_metrics)
        
        overall_precision = total_matched / total_predictions if total_predictions > 0 else 0.0
        overall_recall = total_matched / total_ground_truth if total_ground_truth > 0 else 0.0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        
        # Average metrics
        avg_precision = np.mean([m['precision'] for m in all_metrics])
        avg_recall = np.mean([m['recall'] for m in all_metrics])
        avg_f1 = np.mean([m['f1'] for m in all_metrics])
        avg_iou = np.mean([m['avg_iou'] for m in all_metrics if m['avg_iou'] > 0])
        avg_rmse = np.mean([m['rmse'] for m in all_metrics if m['rmse'] > 0])
        
        print("\n" + "="*60)
        print("📊 OVERALL TEST RESULTS")
        print("="*60)
        print(f"🎯 AGGREGATE METRICS (across all images):")
        print(f"   Total Matched: {total_matched}")
        print(f"   Total Predictions: {total_predictions}")
        print(f"   Total Ground Truth: {total_ground_truth}")
        print(f"   Overall Precision: {overall_precision:.3f}")
        print(f"   Overall Recall: {overall_recall:.3f}")
        print(f"   Overall F1-Score: {overall_f1:.3f}")
        
        print(f"\n📈 AVERAGE METRICS (per image):")
        print(f"   Avg Precision: {avg_precision:.3f}")
        print(f"   Avg Recall: {avg_recall:.3f}")
        print(f"   Avg F1-Score: {avg_f1:.3f}")
        print(f"   Avg IoU: {avg_iou:.3f}")
        print(f"   Avg RMSE: {avg_rmse:.1f} pixels")
        
        print(f"\n🏆 MODEL PERFORMANCE:")
        if overall_recall >= 0.95:
            print("   ✅ EXCELLENT recall - catches almost all objects!")
        elif overall_recall >= 0.80:
            print("   ✅ GOOD recall - catches most objects")
        else:
            print("   ⚠️  Room for improvement in recall")
            
        if overall_precision >= 0.80:
            print("   ✅ EXCELLENT precision - very few false positives!")
        elif overall_precision >= 0.60:
            print("   ✅ GOOD precision - reasonable false positive rate")
        else:
            print("   ⚠️  Room for improvement in precision")
            
        if avg_rmse <= 50:
            print("   ✅ EXCELLENT bounding box accuracy!")
        elif avg_rmse <= 100:
            print("   ✅ GOOD bounding box accuracy")
        else:
            print("   ⚠️  Room for improvement in bounding box accuracy")
    
    def create_summary_visualization(self, all_metrics: List[Dict[str, float]], output_files: List[Path]):
        """Create a summary visualization of all test results."""
        if len(output_files) < 2:
            return
        
        # Create a grid of prediction images
        fig = plt.figure(figsize=(20, 15))
        
        num_images = len(output_files)
        cols = min(3, num_images)
        rows = (num_images + cols - 1) // cols
        
        for i, output_file in enumerate(output_files[:9]):  # Max 9 images
            ax = fig.add_subplot(rows, cols, i + 1)
            
            # Load and display the prediction image
            pred_img = cv2.imread(str(output_file))
            if pred_img is not None:
                pred_img_rgb = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
                ax.imshow(pred_img_rgb)
                ax.set_title(f"{output_file.stem}\nF1: {all_metrics[i]['f1']:.3f}", fontsize=10)
                ax.axis('off')
        
        plt.tight_layout()
        summary_file = "output/test_predictions/summary_grid.png"
        plt.savefig(summary_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📁 Summary grid saved: {summary_file}")


def main():
    """Main testing function."""
    # Model configuration
    model_path = "models/weights/precision_focused.pt"
    data_yaml = "data/augmented_training/data.yaml"
    conf_threshold = 0.2
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please ensure the precision-focused model has been trained and saved.")
        return
    
    # Initialize tester
    tester = PrecisionModelTester(model_path, data_yaml, conf_threshold)
    
    # Test on validation split
    print("🚀 Starting Precision Model Testing")
    print("="*50)
    
    all_metrics, output_files = tester.test_on_dataset_split('val', max_images=5)
    
    print(f"\n✅ Testing complete!")
    print(f"📁 Visual results saved in: output/test_predictions/")
    
    # Also run full evaluation for comparison
    print(f"\n🔍 Running full dataset evaluation for comparison...")
    try:
        from evaluate_model import ModelEvaluator
        evaluator = ModelEvaluator(model_path, data_yaml, conf_threshold=conf_threshold)
        full_metrics = evaluator.evaluate_dataset('val')
        
        print(f"\n📊 Full Dataset Metrics:")
        print(f"   RMSE: {full_metrics.rmse_pixels:.1f}px")
        print(f"   Precision: {full_metrics.precision:.3f}")
        print(f"   Recall: {full_metrics.recall:.3f}")
        print(f"   F1-Score: {full_metrics.f1_score:.3f}")
        
    except Exception as e:
        print(f"⚠️  Could not run full evaluation: {e}")


if __name__ == "__main__":
    main() 