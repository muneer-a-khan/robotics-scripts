"""
Enhanced Orientation Detection System
Provides more accurate component orientation and polarity detection for circuit validation.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math
from enum import Enum

from data_structures import ComponentDetection, ComponentType, BoundingBox


class OrientationResult(Enum):
    """Orientation detection results."""
    CORRECT = "correct"
    INCORRECT = "incorrect"
    REVERSED = "reversed"
    UNKNOWN = "unknown"


@dataclass
class OrientationDetection:
    """Result of orientation detection."""
    component_id: str
    detected_angle: float  # 0-360 degrees
    expected_angle: Optional[float]  # Expected angle if known
    polarity_correct: Optional[bool]  # For polarity-sensitive components
    confidence: float  # 0-1 confidence in detection
    result: OrientationResult


class EnhancedOrientationDetector:
    """
    Enhanced orientation detector for Snap Circuit components.
    Uses computer vision techniques to detect component orientations and polarity.
    """
    
    def __init__(self):
        self.component_orientation_handlers = {
            ComponentType.LED: self._detect_led_orientation,
            ComponentType.BATTERY_HOLDER: self._detect_battery_holder_orientation,
            ComponentType.SWITCH: self._detect_switch_orientation,
            ComponentType.BUTTON: self._detect_button_orientation,
            ComponentType.RESISTOR: self._detect_resistor_orientation,
            ComponentType.MOTOR: self._detect_motor_orientation,
            ComponentType.SPEAKER: self._detect_speaker_orientation,
        }
        
        # Expected orientations for different component types
        self.expected_orientations = {
            ComponentType.LED: [0, 180],  # Forward or reverse
            ComponentType.BATTERY_HOLDER: [0, 180],  # + on left or right
            ComponentType.SWITCH: [0, 90, 180, 270],  # 4-way orientation
            ComponentType.RESISTOR: [0, 90],  # Horizontal or vertical
        }
    
    def detect_orientation(self, component: ComponentDetection, 
                          image: np.ndarray) -> OrientationDetection:
        """
        Detect the orientation of a component in an image.
        
        Args:
            component: The component to analyze
            image: The image containing the component
            
        Returns:
            OrientationDetection result
        """
        # Extract component region from image
        bbox = component.bbox
        x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        component_image = image[y1:y2, x1:x2]
        
        if component_image.size == 0:
            return OrientationDetection(
                component_id=component.id,
                detected_angle=0.0,
                expected_angle=None,
                polarity_correct=None,
                confidence=0.0,
                result=OrientationResult.UNKNOWN
            )
        
        # Use component-specific orientation detection
        if component.component_type in self.component_orientation_handlers:
            return self.component_orientation_handlers[component.component_type](
                component, component_image
            )
        else:
            # Generic orientation detection
            return self._detect_generic_orientation(component, component_image)
    
    def _detect_led_orientation(self, component: ComponentDetection, 
                              component_image: np.ndarray) -> OrientationDetection:
        """Detect LED orientation and polarity."""
        # Convert to grayscale for processing
        gray = cv2.cvtColor(component_image, cv2.COLOR_BGR2GRAY)
        
        # Try to detect LED polarity markers
        polarity_correct = self._detect_led_polarity(gray)
        
        # Detect orientation based on LED shape
        angle = self._detect_component_angle(gray)
        
        # LED-specific validation
        expected_angles = self.expected_orientations.get(ComponentType.LED, [0, 180])
        closest_expected = min(expected_angles, key=lambda x: min(abs(angle - x), abs(angle - x + 360), abs(angle - x - 360)))
        
        angle_diff = min(abs(angle - closest_expected), 
                        abs(angle - closest_expected + 360), 
                        abs(angle - closest_expected - 360))
        
        # Determine result
        if angle_diff < 15:  # Within 15 degrees
            if polarity_correct is True:
                result = OrientationResult.CORRECT
            elif polarity_correct is False:
                result = OrientationResult.REVERSED
            else:
                result = OrientationResult.UNKNOWN
        else:
            result = OrientationResult.INCORRECT
        
        return OrientationDetection(
            component_id=component.id,
            detected_angle=angle,
            expected_angle=closest_expected,
            polarity_correct=polarity_correct,
            confidence=0.8 if angle_diff < 15 else 0.3,
            result=result
        )
    
    def _detect_battery_holder_orientation(self, component: ComponentDetection, 
                                         component_image: np.ndarray) -> OrientationDetection:
        """Detect battery holder orientation and polarity."""
        gray = cv2.cvtColor(component_image, cv2.COLOR_BGR2GRAY)
        
        # Look for + and - symbols
        polarity_correct = self._detect_battery_polarity(gray)
        
        # Detect orientation based on battery holder shape
        angle = self._detect_component_angle(gray)
        
        expected_angles = self.expected_orientations.get(ComponentType.BATTERY_HOLDER, [0, 180])
        closest_expected = min(expected_angles, key=lambda x: min(abs(angle - x), abs(angle - x + 360), abs(angle - x - 360)))
        
        angle_diff = min(abs(angle - closest_expected), 
                        abs(angle - closest_expected + 360), 
                        abs(angle - closest_expected - 360))
        
        if angle_diff < 20:  # Battery holders have more tolerance
            result = OrientationResult.CORRECT if polarity_correct else OrientationResult.REVERSED
        else:
            result = OrientationResult.INCORRECT
        
        return OrientationDetection(
            component_id=component.id,
            detected_angle=angle,
            expected_angle=closest_expected,
            polarity_correct=polarity_correct,
            confidence=0.7 if angle_diff < 20 else 0.2,
            result=result
        )
    
    def _detect_switch_orientation(self, component: ComponentDetection, 
                                 component_image: np.ndarray) -> OrientationDetection:
        """Detect switch orientation."""
        gray = cv2.cvtColor(component_image, cv2.COLOR_BGR2GRAY)
        
        # Detect switch lever position
        angle = self._detect_switch_lever_angle(gray)
        
        expected_angles = self.expected_orientations.get(ComponentType.SWITCH, [0, 90, 180, 270])
        closest_expected = min(expected_angles, key=lambda x: min(abs(angle - x), abs(angle - x + 360), abs(angle - x - 360)))
        
        angle_diff = min(abs(angle - closest_expected), 
                        abs(angle - closest_expected + 360), 
                        abs(angle - closest_expected - 360))
        
        result = OrientationResult.CORRECT if angle_diff < 30 else OrientationResult.INCORRECT
        
        return OrientationDetection(
            component_id=component.id,
            detected_angle=angle,
            expected_angle=closest_expected,
            polarity_correct=None,  # Switches don't have polarity
            confidence=0.6 if angle_diff < 30 else 0.3,
            result=result
        )
    
    def _detect_button_orientation(self, component: ComponentDetection, 
                                 component_image: np.ndarray) -> OrientationDetection:
        """Detect button orientation."""
        # Buttons are typically circular, so orientation matters less
        gray = cv2.cvtColor(component_image, cv2.COLOR_BGR2GRAY)
        angle = self._detect_component_angle(gray)
        
        return OrientationDetection(
            component_id=component.id,
            detected_angle=angle,
            expected_angle=0,  # Buttons don't have strict orientation
            polarity_correct=None,
            confidence=0.5,  # Lower confidence since orientation is less important
            result=OrientationResult.CORRECT
        )
    
    def _detect_resistor_orientation(self, component: ComponentDetection, 
                                   component_image: np.ndarray) -> OrientationDetection:
        """Detect resistor orientation."""
        gray = cv2.cvtColor(component_image, cv2.COLOR_BGR2GRAY)
        
        # Resistors are typically rectangular
        angle = self._detect_component_angle(gray)
        
        expected_angles = self.expected_orientations.get(ComponentType.RESISTOR, [0, 90])
        closest_expected = min(expected_angles, key=lambda x: min(abs(angle - x), abs(angle - x + 360), abs(angle - x - 360)))
        
        angle_diff = min(abs(angle - closest_expected), 
                        abs(angle - closest_expected + 360), 
                        abs(angle - closest_expected - 360))
        
        result = OrientationResult.CORRECT if angle_diff < 20 else OrientationResult.INCORRECT
        
        return OrientationDetection(
            component_id=component.id,
            detected_angle=angle,
            expected_angle=closest_expected,
            polarity_correct=None,  # Resistors don't have polarity
            confidence=0.7 if angle_diff < 20 else 0.3,
            result=result
        )
    
    def _detect_motor_orientation(self, component: ComponentDetection, 
                                component_image: np.ndarray) -> OrientationDetection:
        """Detect motor orientation."""
        gray = cv2.cvtColor(component_image, cv2.COLOR_BGR2GRAY)
        
        # Motors may have polarity sensitivity
        polarity_correct = self._detect_motor_polarity(gray)
        angle = self._detect_component_angle(gray)
        
        # Motors typically have some preferred orientations
        expected_angles = [0, 90, 180, 270]
        closest_expected = min(expected_angles, key=lambda x: min(abs(angle - x), abs(angle - x + 360), abs(angle - x - 360)))
        
        angle_diff = min(abs(angle - closest_expected), 
                        abs(angle - closest_expected + 360), 
                        abs(angle - closest_expected - 360))
        
        result = OrientationResult.CORRECT if angle_diff < 25 else OrientationResult.INCORRECT
        
        return OrientationDetection(
            component_id=component.id,
            detected_angle=angle,
            expected_angle=closest_expected,
            polarity_correct=polarity_correct,
            confidence=0.6 if angle_diff < 25 else 0.3,
            result=result
        )
    
    def _detect_speaker_orientation(self, component: ComponentDetection, 
                                  component_image: np.ndarray) -> OrientationDetection:
        """Detect speaker orientation."""
        gray = cv2.cvtColor(component_image, cv2.COLOR_BGR2GRAY)
        
        # Speakers may have polarity
        polarity_correct = self._detect_speaker_polarity(gray)
        angle = self._detect_component_angle(gray)
        
        # Speakers are often circular but may have preferred orientations
        expected_angles = [0, 90, 180, 270]
        closest_expected = min(expected_angles, key=lambda x: min(abs(angle - x), abs(angle - x + 360), abs(angle - x - 360)))
        
        angle_diff = min(abs(angle - closest_expected), 
                        abs(angle - closest_expected + 360), 
                        abs(angle - closest_expected - 360))
        
        result = OrientationResult.CORRECT if angle_diff < 30 else OrientationResult.INCORRECT
        
        return OrientationDetection(
            component_id=component.id,
            detected_angle=angle,
            expected_angle=closest_expected,
            polarity_correct=polarity_correct,
            confidence=0.5 if angle_diff < 30 else 0.3,
            result=result
        )
    
    def _detect_generic_orientation(self, component: ComponentDetection, 
                                  component_image: np.ndarray) -> OrientationDetection:
        """Generic orientation detection for unknown component types."""
        gray = cv2.cvtColor(component_image, cv2.COLOR_BGR2GRAY)
        angle = self._detect_component_angle(gray)
        
        return OrientationDetection(
            component_id=component.id,
            detected_angle=angle,
            expected_angle=None,
            polarity_correct=None,
            confidence=0.3,  # Low confidence for generic detection
            result=OrientationResult.UNKNOWN
        )
    
    def _detect_component_angle(self, gray_image: np.ndarray) -> float:
        """Detect component angle using edge detection and line fitting."""
        # Edge detection
        edges = cv2.Canny(gray_image, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Fit a minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        
        # Normalize angle to 0-360 range
        if angle < 0:
            angle += 90
        
        return angle
    
    def _detect_switch_lever_angle(self, gray_image: np.ndarray) -> float:
        """Detect switch lever angle specifically."""
        # Use HoughLines to detect the switch lever
        edges = cv2.Canny(gray_image, 50, 150)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=10, maxLineGap=5)
        
        if lines is not None and len(lines) > 0:
            # Find the longest line (likely the switch lever)
            longest_line = max(lines, key=lambda line: 
                              np.sqrt((line[0][2] - line[0][0])**2 + (line[0][3] - line[0][1])**2))
            
            x1, y1, x2, y2 = longest_line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            
            # Normalize to 0-360
            if angle < 0:
                angle += 360
            
            return angle
        
        return 0.0
    
    def _detect_led_polarity(self, gray_image: np.ndarray) -> Optional[bool]:
        """Detect LED polarity using image analysis."""
        # Look for polarity markers (+ and - symbols or cathode/anode indicators)
        # This is a simplified implementation - real polarity detection would be more complex
        
        # Try to find bright spots (could be polarity markers)
        _, thresh = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours that might be polarity markers
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze contour shapes to determine polarity
        # This is simplified - real implementation would need training data
        
        return None  # Return None if polarity can't be determined
    
    def _detect_battery_polarity(self, gray_image: np.ndarray) -> Optional[bool]:
        """Detect battery holder polarity."""
        # Look for + and - symbols
        # This would require template matching or OCR for real implementation
        
        return None  # Simplified - return None if can't determine
    
    def _detect_motor_polarity(self, gray_image: np.ndarray) -> Optional[bool]:
        """Detect motor polarity."""
        # Look for + and - terminals on motor
        
        return None  # Simplified implementation
    
    def _detect_speaker_polarity(self, gray_image: np.ndarray) -> Optional[bool]:
        """Detect speaker polarity."""
        # Look for + and - terminals on speaker
        
        return None  # Simplified implementation
    
    def validate_all_orientations(self, components: List[ComponentDetection], 
                                 image: np.ndarray) -> List[OrientationDetection]:
        """
        Validate orientations for all components in an image.
        
        Args:
            components: List of detected components
            image: The image containing the components
            
        Returns:
            List of orientation detection results
        """
        results = []
        
        for component in components:
            orientation_result = self.detect_orientation(component, image)
            results.append(orientation_result)
        
        return results
    
    def get_orientation_issues(self, orientation_results: List[OrientationDetection]) -> List[Dict[str, Any]]:
        """
        Extract orientation issues from detection results.
        
        Args:
            orientation_results: List of orientation detection results
            
        Returns:
            List of orientation issues for the validator
        """
        issues = []
        
        for result in orientation_results:
            if result.result == OrientationResult.INCORRECT:
                issues.append({
                    "type": "orientation",
                    "component_id": result.component_id,
                    "severity": "error",
                    "message": f"Component orientation is incorrect (detected: {result.detected_angle:.1f}°, expected: {result.expected_angle:.1f}°)",
                    "suggestion": f"Rotate component to {result.expected_angle:.1f}° orientation"
                })
            elif result.result == OrientationResult.REVERSED:
                issues.append({
                    "type": "orientation",
                    "component_id": result.component_id,
                    "severity": "error",
                    "message": "Component polarity is reversed",
                    "suggestion": "Flip component to correct polarity"
                })
            elif result.result == OrientationResult.UNKNOWN and result.confidence < 0.5:
                issues.append({
                    "type": "orientation",
                    "component_id": result.component_id,
                    "severity": "warning",
                    "message": "Component orientation could not be determined",
                    "suggestion": "Check component placement and orientation manually"
                })
        
        return issues 