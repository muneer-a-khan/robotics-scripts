#!/usr/bin/env python3
"""
Circuit Feedback Window System
Shows real-time feedback about circuit building progress, missing components, and hand detection
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Set, Optional
import json

@dataclass
class CircuitFeedback:
    """Data structure for circuit feedback information"""
    added_components: List[str]
    missing_components: List[str]
    present_components: List[str]
    connection_issues: List[str]
    circuit_closed: bool
    hand_detected: bool
    completion_percentage: float
    suggestions: List[str]

class CircuitFeedbackWindow:
    """Real-time feedback window for circuit building"""
    
    def __init__(self):
        self.root = None
        self.window_thread = None
        self.running = False
        self.current_feedback = None
        
        # UI Elements
        self.added_text = None
        self.missing_text = None
        self.present_text = None
        self.status_text = None
        self.progress_bar = None
        self.hand_warning = None
        self.suggestions_text = None
        
        # Colors
        self.colors = {
            'success': '#4CAF50',      # Green
            'warning': '#FF9800',      # Orange  
            'error': '#F44336',        # Red
            'info': '#2196F3',         # Blue
            'hand_warning': '#E91E63'  # Pink/Red
        }
        
    def start_window(self):
        """Start the feedback window in a separate thread"""
        if not self.running:
            try:
                self.running = True
                self.window_thread = threading.Thread(target=self._run_window, daemon=True)
                self.window_thread.start()
                time.sleep(0.5)  # Give window time to initialize
            except Exception as e:
                print(f"❌ Failed to start feedback window: {e}")
                print("💡 This may be due to tkinter compatibility issues on your system")
                self.running = False
                raise
            
    def stop_window(self):
        """Stop the feedback window"""
        self.running = False
        if self.root:
            try:
                self.root.quit()
                self.root.destroy()
            except:
                pass
                
    def _run_window(self):
        """Run the main window loop"""
        self.root = tk.Tk()
        self.root.title("🔧 Circuit Building Feedback")
        self.root.geometry("500x700")
        self.root.configure(bg='#f0f0f0')
        
        # Make window stay on top but not always on top
        self.root.attributes('-topmost', True)
        self.root.after(100, lambda: self.root.attributes('-topmost', False))
        
        self._create_widgets()
        
        # Update loop
        self.root.after(100, self._update_display)
        
        try:
            self.root.mainloop()
        except:
            pass
        finally:
            self.running = False
            
    def _create_widgets(self):
        """Create all UI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_frame, text="🎯 Circuit Building Progress", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=(0, 10))
        
        # Progress bar
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(progress_frame, text="Completion:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        self.progress_bar = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))
        
        # Hand detection warning
        self.hand_warning = tk.Label(main_frame, text="", font=('Arial', 12, 'bold'),
                                   fg='white', bg=self.colors['hand_warning'], pady=5)
        
        # Added components (Success)
        added_frame = self._create_section_frame(main_frame, "✅ Successfully Added", self.colors['success'])
        self.added_text = self._create_text_widget(added_frame, height=3)
        
        # Missing components (Error)  
        missing_frame = self._create_section_frame(main_frame, "❌ Missing Components", self.colors['error'])
        self.missing_text = self._create_text_widget(missing_frame, height=3)
        
        # Present components (Info)
        present_frame = self._create_section_frame(main_frame, "🔍 Currently Detected", self.colors['info'])
        self.present_text = self._create_text_widget(present_frame, height=3)
        
        # Circuit status
        status_frame = self._create_section_frame(main_frame, "⚡ Circuit Status", self.colors['warning'])
        self.status_text = self._create_text_widget(status_frame, height=3)
        
        # Suggestions
        suggestions_frame = self._create_section_frame(main_frame, "💡 Next Steps", self.colors['info'])
        self.suggestions_text = self._create_text_widget(suggestions_frame, height=4)
        
    def _create_section_frame(self, parent, title, color):
        """Create a labeled section frame"""
        frame = ttk.LabelFrame(parent, text=title, padding=10)
        frame.pack(fill=tk.BOTH, expand=True, pady=5)
        return frame
        
    def _create_text_widget(self, parent, height=3):
        """Create a text widget for displaying information"""
        text_widget = scrolledtext.ScrolledText(parent, height=height, wrap=tk.WORD,
                                               font=('Courier', 10), state=tk.DISABLED)
        text_widget.pack(fill=tk.BOTH, expand=True)
        return text_widget
        
    def _update_text_widget(self, widget, content, color='black'):
        """Update text widget content"""
        if widget and self.running:
            try:
                widget.config(state=tk.NORMAL)
                widget.delete(1.0, tk.END)
                widget.insert(1.0, content)
                widget.config(state=tk.DISABLED, fg=color)
            except:
                pass
                
    def _update_display(self):
        """Update the display with current feedback"""
        if not self.running:
            return
            
        if self.current_feedback:
            feedback = self.current_feedback
            
            # Update progress bar
            if self.progress_bar:
                try:
                    self.progress_bar['value'] = feedback.completion_percentage
                    progress_text = f"{feedback.completion_percentage:.1f}% Complete"
                    self.root.title(f"🔧 Circuit Building Feedback - {progress_text}")
                except:
                    pass
            
            # Hand detection warning
            if feedback.hand_detected:
                self.hand_warning.config(text="⚠️ HAND ON BOARD! ⚠️")
                self.hand_warning.pack(fill=tk.X, pady=5)
            else:
                self.hand_warning.pack_forget()
            
            # Added components
            added_content = ""
            for component in feedback.added_components:
                added_content += f"✓ {component}\n"
            if not added_content:
                added_content = "No components added yet..."
            self._update_text_widget(self.added_text, added_content, self.colors['success'])
            
            # Missing components
            missing_content = ""
            for component in feedback.missing_components:
                missing_content += f"• The following is NOT in the circuit: {component}\n"
            if not missing_content:
                missing_content = "All required components detected! ✅"
            self._update_text_widget(self.missing_text, missing_content, self.colors['error'])
            
            # Present components
            present_content = ""
            for component in feedback.present_components:
                present_content += f"• The following exists in the circuit: {component}\n"
            if not present_content:
                present_content = "No components detected..."
            self._update_text_widget(self.present_text, present_content, self.colors['info'])
            
            # Circuit status
            status_content = ""
            for issue in feedback.connection_issues:
                status_content += f"⚠️ {issue}\n"
            
            if feedback.circuit_closed:
                status_content += "✅ Circuit CLOSED\n"
            else:
                status_content += "❌ Circuit NOT closed\n"
                
            if not status_content.strip():
                status_content = "Analyzing circuit connections..."
            self._update_text_widget(self.status_text, status_content)
            
            # Suggestions
            suggestions_content = ""
            for suggestion in feedback.suggestions:
                suggestions_content += f"💡 {suggestion}\n"
            if not suggestions_content:
                suggestions_content = "Keep building your circuit! 🔧"
            self._update_text_widget(self.suggestions_text, suggestions_content, self.colors['info'])
        
        # Schedule next update
        if self.running:
            self.root.after(500, self._update_display)  # Update every 500ms
            
    def update_feedback(self, feedback: CircuitFeedback):
        """Update the feedback data"""
        self.current_feedback = feedback
        
    def is_running(self):
        """Check if window is running"""
        return self.running and self.root is not None


class CircuitAnalyzer:
    """Analyzes circuit progress and generates feedback"""
    
    def __init__(self):
        self.hand_detection_threshold = 0.3  # Confidence threshold for hand detection
        
    def analyze_circuit_progress(self, current_detections: Dict, ground_truth: Dict, 
                                model_names: Dict) -> CircuitFeedback:
        """Analyze current circuit against ground truth and generate feedback"""
        
        # Extract component lists
        current_components = self._extract_components(current_detections, model_names)
        target_components = self._extract_ground_truth_components(ground_truth)
        
        # Determine what's been added, what's missing
        added_components = []
        missing_components = []
        present_components = list(current_components.keys())
        
        # Check each target component
        for target_comp, target_count in target_components.items():
            current_count = current_components.get(target_comp, 0)
            
            if current_count >= target_count:
                added_components.append(f"{target_comp} ({current_count}/{target_count})")
            else:
                missing_components.append(f"{target_comp} (need {target_count - current_count} more)")
                
        # Check for hand detection
        hand_detected = self._detect_hands(current_detections, model_names)
        
        # Analyze connections (simplified)
        connection_issues = self._analyze_connections(current_detections)
        
        # Calculate completion percentage
        completion_percentage = self._calculate_completion(current_components, target_components)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(current_components, target_components, missing_components)
        
        # Circuit closure analysis (simplified)
        circuit_closed = len(missing_components) == 0 and len(connection_issues) == 0
        
        return CircuitFeedback(
            added_components=added_components,
            missing_components=missing_components,
            present_components=present_components,
            connection_issues=connection_issues,
            circuit_closed=circuit_closed,
            hand_detected=hand_detected,
            completion_percentage=completion_percentage,
            suggestions=suggestions
        )
        
    def _extract_components(self, detections: Dict, model_names: Dict) -> Dict[str, int]:
        """Extract component counts from current detections"""
        components = {}
        
        # Combine left and right detections
        all_detections = {}
        if 'left' in detections:
            all_detections.update(detections.get('left', {}))
        if 'right' in detections:
            for comp, dets in detections.get('right', {}).items():
                if comp in all_detections:
                    all_detections[comp].extend(dets)
                else:
                    all_detections[comp] = dets
        
        # Count each component type
        for comp_type, detection_list in all_detections.items():
            if comp_type != "Green tape":  # Ignore green tape
                components[comp_type] = len(detection_list)
                
        return components
        
    def _extract_ground_truth_components(self, ground_truth: Dict) -> Dict[str, int]:
        """Extract required components from ground truth"""
        components = {}
        
        if 'left_detections' in ground_truth:
            for comp_type, detection_list in ground_truth['left_detections'].items():
                if comp_type != "Green tape":  # Ignore green tape
                    components[comp_type] = len(detection_list)
                    
        return components
        
    def _detect_hands(self, detections: Dict, model_names: Dict) -> bool:
        """Detect if hands are present on the board"""
        # For now, this is a placeholder - you could train your model to detect hands
        # or use a separate hand detection model
        
        # Simple heuristic: if there are many detections with low confidence, might be hands
        all_detections = {}
        if 'left' in detections:
            all_detections.update(detections.get('left', {}))
        if 'right' in detections:
            for comp, dets in detections.get('right', {}).items():
                if comp in all_detections:
                    all_detections[comp].extend(dets)
                else:
                    all_detections[comp] = dets
        
        # Count low-confidence detections
        low_confidence_count = 0
        total_detections = 0
        
        for comp_type, detection_list in all_detections.items():
            for detection in detection_list:
                total_detections += 1
                if isinstance(detection, dict) and 'confidence' in detection:
                    if detection['confidence'] < 0.4:  # Low confidence might indicate occlusion
                        low_confidence_count += 1
                        
        # If more than 30% of detections are low confidence, might be hands
        if total_detections > 0:
            low_confidence_ratio = low_confidence_count / total_detections
            return low_confidence_ratio > 0.3
            
        return False
        
    def _analyze_connections(self, detections: Dict) -> List[str]:
        """Analyze circuit connections and identify issues"""
        issues = []
        
        # Count total components (excluding green tape)
        total_components = 0
        all_detections = {}
        
        if 'left' in detections:
            all_detections.update(detections.get('left', {}))
        if 'right' in detections:
            for comp, dets in detections.get('right', {}).items():
                if comp in all_detections:
                    all_detections[comp].extend(dets)
                else:
                    all_detections[comp] = dets
        
        for comp_type, detection_list in all_detections.items():
            if comp_type != "Green tape":
                total_components += len(detection_list)
        
        # Simple connection analysis
        wire_count = len(all_detections.get("Wire", []))
        component_count = total_components - wire_count
        
        if component_count > 0:
            # Need at least (components - 1) wires for basic connectivity
            min_wires_needed = max(1, component_count - 1)
            if wire_count < min_wires_needed:
                issues.append(f"Too many pieces were not connected to any others")
                issues.append(f"Need at least {min_wires_needed} wires, found {wire_count}")
        
        return issues
        
    def _calculate_completion(self, current: Dict[str, int], target: Dict[str, int]) -> float:
        """Calculate completion percentage"""
        if not target:
            return 100.0
            
        total_required = sum(target.values())
        total_current = 0
        
        for comp, required_count in target.items():
            current_count = current.get(comp, 0)
            total_current += min(current_count, required_count)
            
        return (total_current / total_required) * 100.0 if total_required > 0 else 0.0
        
    def _generate_suggestions(self, current: Dict[str, int], target: Dict[str, int], 
                            missing: List[str]) -> List[str]:
        """Generate helpful suggestions for next steps"""
        suggestions = []
        
        if missing:
            suggestions.append(f"Add missing components: {', '.join([m.split('(')[0].strip() for m in missing[:3]])}")
        
        # Check for wires
        current_wires = current.get("Wire", 0)
        target_wires = target.get("Wire", 0)
        
        if current_wires < target_wires:
            suggestions.append(f"Connect components with {target_wires - current_wires} more wire(s)")
        
        # Generic suggestions based on completion
        completion = self._calculate_completion(current, target)
        if completion < 30:
            suggestions.append("Start by placing the main components (battery, switch, etc.)")
        elif completion < 70:
            suggestions.append("Connect your components with wires")
        elif completion < 100:
            suggestions.append("Almost there! Check all connections")
        else:
            suggestions.append("Circuit complete! Great job! 🎉")
            
        return suggestions


# Integration functions for the main system
def create_feedback_system():
    """Create and return a new feedback system"""
    return CircuitFeedbackWindow(), CircuitAnalyzer()

if __name__ == "__main__":
    # Demo/test the feedback window
    import random
    
    window, analyzer = create_feedback_system()
    window.start_window()
    
    # Simulate feedback updates
    try:
        for i in range(20):
            # Generate sample feedback
            feedback = CircuitFeedback(
                added_components=[f"Battery ({random.randint(1,2)}/1)", "Wire (2/3)"] if i > 5 else [],
                missing_components=["Switch", "Lamp"] if i < 15 else [],
                present_components=["Battery", "Wire", "Green tape"],
                connection_issues=["Too many pieces were not connected to any others"] if i < 12 else [],
                circuit_closed=i > 15,
                hand_detected=random.random() < 0.3,  # 30% chance
                completion_percentage=min(100, i * 5 + random.randint(-10, 10)),
                suggestions=["Add a switch to control the circuit", "Connect battery to other components"]
            )
            
            window.update_feedback(feedback)
            time.sleep(2)
            
    except KeyboardInterrupt:
        pass
    finally:
        window.stop_window()
