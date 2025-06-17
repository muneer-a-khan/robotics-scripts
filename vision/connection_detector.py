"""
OpenCV-based connection detector for finding wire paths and connections between components.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
import networkx as nx

from config import CONNECTION_CONFIG
from data_structures import ComponentDetection, Connection, BoundingBox


class ConnectionDetector:
    """
    Detects connections between Snap Circuit components using OpenCV.
    """
    
    def __init__(self):
        """Initialize the connection detector."""
        self.config = CONNECTION_CONFIG
        self.wire_color_lower = np.array(self.config["wire_color_range"]["lower"])
        self.wire_color_upper = np.array(self.config["wire_color_range"]["upper"])
        self.min_contour_area = self.config["min_contour_area"]
        self.max_contour_area = self.config["max_contour_area"]
        self.proximity_threshold = self.config["connection_proximity_threshold"]
        self.line_thickness = self.config["line_thickness"]
        
    def detect_connections(self, image: np.ndarray, 
                          components: List[ComponentDetection]) -> List[Connection]:
        """
        Detect connections between components in the image.
        
        Args:
            image: Input image (BGR format)
            components: List of detected components
            
        Returns:
            List of detected connections
        """
        if len(components) < 2:
            return []
        
        # Extract wire paths
        wire_mask = self._extract_wire_mask(image)
        wire_paths = self._find_wire_paths(wire_mask)
        
        # Find connection points for each component
        component_connection_points = self._find_component_connection_points(
            components, wire_mask
        )
        
        # Build connections based on wire paths and proximity
        connections = self._build_connections(
            components, wire_paths, component_connection_points
        )
        
        return connections
    
    def _extract_wire_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Extract connection regions from the image using color segmentation optimized for Snap Circuits.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Binary mask of connection regions
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for metallic connection points (silver/gold snap points)
        if "metallic_color_range" in self.config:
            metallic_lower = np.array(self.config["metallic_color_range"]["lower"])
            metallic_upper = np.array(self.config["metallic_color_range"]["upper"])
            metallic_mask = cv2.inRange(hsv, metallic_lower, metallic_upper)
        else:
            metallic_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        # Create mask for general connection colors
        wire_mask = cv2.inRange(hsv, self.wire_color_lower, self.wire_color_upper)
        
        # Edge detection for fine details
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)  # Lower thresholds for Snap Circuit details
        
        # Dilate edges slightly
        kernel = np.ones((2, 2), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Combine all masks
        combined_mask = cv2.bitwise_or(wire_mask, metallic_mask)
        combined_mask = cv2.bitwise_or(combined_mask, edges_dilated)
        
        # Clean up the mask
        combined_mask = self._clean_wire_mask(combined_mask)
        
        return combined_mask
    
    def _clean_wire_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Clean up the wire mask by removing noise and small objects.
        
        Args:
            mask: Binary mask to clean
            
        Returns:
            Cleaned binary mask
        """
        # Remove small noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Close gaps in wires
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove very small or very large contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cleaned_mask = np.zeros_like(mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area <= area <= self.max_contour_area:
                cv2.drawContours(cleaned_mask, [contour], -1, 255, -1)
        
        return cleaned_mask
    
    def _find_wire_paths(self, wire_mask: np.ndarray) -> List[List[Tuple[int, int]]]:
        """
        Find wire paths by skeletonizing and tracing contours.
        
        Args:
            wire_mask: Binary mask of wire regions
            
        Returns:
            List of wire paths, each path is a list of (x, y) points
        """
        # Skeletonize to get centerlines
        skeleton = skeletonize(wire_mask > 0)
        skeleton = (skeleton * 255).astype(np.uint8)
        
        # Find contours of skeleton
        contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        wire_paths = []
        for contour in contours:
            if len(contour) > 10:  # Minimum length for a valid wire
                # Convert contour to list of points
                path = [(point[0][0], point[0][1]) for point in contour]
                wire_paths.append(path)
        
        return wire_paths
    
    def _find_component_connection_points(self, 
                                        components: List[ComponentDetection],
                                        wire_mask: np.ndarray) -> Dict[str, List[Tuple[int, int]]]:
        """
        Find connection points for each component based on wire intersections.
        
        Args:
            components: List of detected components
            wire_mask: Binary mask of wire regions
            
        Returns:
            Dictionary mapping component IDs to their connection points
        """
        connection_points = {}
        
        for component in components:
            points = []
            bbox = component.bbox
            
            # Define search region around component (using config margin)
            margin = self.config.get("component_margin", 20)
            x1 = max(0, int(bbox.x1 - margin))
            y1 = max(0, int(bbox.y1 - margin))
            x2 = min(wire_mask.shape[1], int(bbox.x2 + margin))
            y2 = min(wire_mask.shape[0], int(bbox.y2 + margin))
            
            # Extract region around component
            region_mask = wire_mask[y1:y2, x1:x2]
            
            # Find points where wires intersect with component boundary
            # Check edges of component bounding box
            edges = [
                (int(bbox.x1), range(int(bbox.y1), int(bbox.y2))),  # Left edge
                (int(bbox.x2), range(int(bbox.y1), int(bbox.y2))),  # Right edge
                (range(int(bbox.x1), int(bbox.x2)), int(bbox.y1)),  # Top edge
                (range(int(bbox.x1), int(bbox.x2)), int(bbox.y2))   # Bottom edge
            ]
            
            for edge in edges:
                if isinstance(edge[0], int):  # Vertical edge
                    x, y_range = edge
                    for y in y_range:
                        if (0 <= x < wire_mask.shape[1] and 
                            0 <= y < wire_mask.shape[0] and 
                            wire_mask[y, x] > 0):
                            points.append((x, y))
                else:  # Horizontal edge
                    x_range, y = edge
                    for x in x_range:
                        if (0 <= x < wire_mask.shape[1] and 
                            0 <= y < wire_mask.shape[0] and 
                            wire_mask[y, x] > 0):
                            points.append((x, y))
            
            # Remove duplicate points
            points = list(set(points))
            connection_points[component.id] = points
            
            # Update component with connection points
            component.connection_points = points
        
        return connection_points
    
    def _build_connections(self, 
                          components: List[ComponentDetection],
                          wire_paths: List[List[Tuple[int, int]]],
                          connection_points: Dict[str, List[Tuple[int, int]]]) -> List[Connection]:
        """
        Build connections between components based on wire paths and proximity.
        
        Args:
            components: List of detected components
            wire_paths: List of wire paths
            connection_points: Component connection points
            
        Returns:
            List of connections
        """
        connections = []
        
        # Method 1: Direct proximity between components
        connections.extend(self._find_proximity_connections(components))
        
        # Method 2: Wire path connections
        connections.extend(self._find_wire_path_connections(
            components, wire_paths, connection_points
        ))
        
        # Remove duplicates
        connections = self._remove_duplicate_connections(connections)
        
        return connections
    
    def _find_proximity_connections(self, 
                                  components: List[ComponentDetection]) -> List[Connection]:
        """
        Find connections based on component proximity.
        
        Args:
            components: List of detected components
            
        Returns:
            List of proximity-based connections
        """
        connections = []
        
        for i, comp1 in enumerate(components):
            for j, comp2 in enumerate(components[i+1:], i+1):
                distance = self._calculate_component_distance(comp1, comp2)
                
                if distance < self.proximity_threshold:
                    connection = Connection(
                        component_id_1=comp1.id,
                        component_id_2=comp2.id,
                        connection_type="proximity",
                        confidence=max(0.0, 1.0 - distance / self.proximity_threshold)
                    )
                    connections.append(connection)
        
        return connections
    
    def _find_wire_path_connections(self,
                                  components: List[ComponentDetection],
                                  wire_paths: List[List[Tuple[int, int]]],
                                  connection_points: Dict[str, List[Tuple[int, int]]]) -> List[Connection]:
        """
        Find connections based on wire paths connecting components.
        
        Args:
            components: List of detected components
            wire_paths: List of wire paths
            connection_points: Component connection points
            
        Returns:
            List of wire-based connections
        """
        connections = []
        
        # Performance safeguard: limit number of paths processed
        max_paths = 50  # Reasonable limit to prevent infinite loops
        paths_to_process = wire_paths[:max_paths]
        
        for path in paths_to_process:
            connected_components = []
            
            # Limit processing for performance - sample path points if too many
            sampled_path = path[::max(1, len(path)//20)] if len(path) > 100 else path
            
            # Find which components this wire path connects
            for component in components:
                comp_points = connection_points.get(component.id, [])
                
                # Check if any component connection points are near this wire path
                for comp_point in comp_points:
                    min_distance = float('inf')
                    for path_point in sampled_path:
                        distance = np.sqrt(
                            (comp_point[0] - path_point[0])**2 + 
                            (comp_point[1] - path_point[1])**2
                        )
                        min_distance = min(min_distance, distance)
                        if distance < self.proximity_threshold:
                            connected_components.append(component)
                            break
                    if component in connected_components:
                        break
            
            # Create connections between components connected by this wire
            for i, comp1 in enumerate(connected_components):
                for comp2 in connected_components[i+1:]:
                    connection = Connection(
                        component_id_1=comp1.id,
                        component_id_2=comp2.id,
                        connection_type="wire",
                        confidence=0.8,
                        path_points=path
                    )
                    connections.append(connection)
        
        return connections
    
    def _calculate_component_distance(self, 
                                    comp1: ComponentDetection, 
                                    comp2: ComponentDetection) -> float:
        """
        Calculate distance between two components.
        
        Args:
            comp1: First component
            comp2: Second component
            
        Returns:
            Distance between component centers
        """
        center1 = comp1.bbox.center
        center2 = comp2.bbox.center
        
        return np.sqrt(
            (center1[0] - center2[0])**2 + 
            (center1[1] - center2[1])**2
        )
    
    def _remove_duplicate_connections(self, connections: List[Connection]) -> List[Connection]:
        """
        Remove duplicate connections.
        
        Args:
            connections: List of connections possibly containing duplicates
            
        Returns:
            List of unique connections
        """
        seen = set()
        unique_connections = []
        
        for connection in connections:
            # Create a normalized tuple for comparison
            key = tuple(sorted([connection.component_id_1, connection.component_id_2]))
            
            if key not in seen:
                seen.add(key)
                unique_connections.append(connection)
        
        return unique_connections
    
    def visualize_connections(self, 
                            image: np.ndarray,
                            components: List[ComponentDetection],
                            connections: List[Connection]) -> np.ndarray:
        """
        Visualize detected connections on the image.
        
        Args:
            image: Input image
            components: List of detected components
            connections: List of detected connections
            
        Returns:
            Image with connections visualized
        """
        vis_image = image.copy()
        
        # Draw components
        for component in components:
            bbox = component.bbox
            cv2.rectangle(
                vis_image,
                (int(bbox.x1), int(bbox.y1)),
                (int(bbox.x2), int(bbox.y2)),
                (0, 255, 0),  # Green for components
                2
            )
            
            # Draw connection points
            for point in component.connection_points:
                cv2.circle(vis_image, point, 3, (255, 0, 0), -1)  # Blue dots
        
        # Draw connections
        component_dict = {comp.id: comp for comp in components}
        
        for connection in connections:
            comp1 = component_dict.get(connection.component_id_1)
            comp2 = component_dict.get(connection.component_id_2)
            
            if comp1 and comp2:
                center1 = comp1.bbox.center
                center2 = comp2.bbox.center
                
                # Draw connection line
                cv2.line(
                    vis_image,
                    (int(center1[0]), int(center1[1])),
                    (int(center2[0]), int(center2[1])),
                    (0, 0, 255),  # Red for connections
                    2
                )
                
                # Draw wire path if available
                if connection.path_points:
                    for i in range(len(connection.path_points) - 1):
                        pt1 = connection.path_points[i]
                        pt2 = connection.path_points[i + 1]
                        cv2.line(vis_image, pt1, pt2, (255, 255, 0), 1)  # Yellow for wire paths
        
        return vis_image


def test_connection_detector():
    """Test function for the connection detector."""
    detector = ConnectionDetector()
    
    # Create a test image with some mock components
    test_image = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Draw some mock wires
    cv2.line(test_image, (100, 100), (200, 100), (128, 128, 128), 3)
    cv2.line(test_image, (200, 100), (200, 200), (128, 128, 128), 3)
    
    # Create mock components
    from data_structures import ComponentType, BoundingBox
    
    components = [
        ComponentDetection(
            id="battery-1",
            label="battery_holder",
            bbox=BoundingBox(80, 80, 120, 120),
            orientation=0,
            confidence=0.9,
            component_type=ComponentType.BATTERY_HOLDER
        ),
        ComponentDetection(
            id="led-1", 
            label="led",
            bbox=BoundingBox(180, 180, 220, 220),
            orientation=0,
            confidence=0.8,
            component_type=ComponentType.LED
        )
    ]
    
    # Detect connections
    connections = detector.detect_connections(test_image, components)
    print(f"Detected {len(connections)} connections")
    
    # Visualize
    vis_image = detector.visualize_connections(test_image, components, connections)
    cv2.imshow("Connection Detection Test", vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_connection_detector() 