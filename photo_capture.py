#!/usr/bin/env python3
"""
Simple camera capture script for collecting training photos.
Shows live camera feed with center line for dual-board alignment.
Press SPACE to capture photos, Q to quit.
"""

import cv2
import os
from datetime import datetime
import numpy as np

def main():
    # Create photos directory if it doesn't exist
    photos_dir = "photos"
    if not os.path.exists(photos_dir):
        os.makedirs(photos_dir)
        print(f"Created directory: {photos_dir}")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)  # Use camera 0 (default)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera resolution (optional - adjust as needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Camera initialized successfully!")
    print("Controls:")
    print("  SPACE - Capture photo")
    print("  Q - Quit")
    print("  ESC - Quit")
    
    photo_count = 0
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        line_x = width // 2 - 30  # Move line 30 pixels left of center
        
        # Draw vertical line slightly left of center
        cv2.line(frame, (line_x, 0), (line_x, height), (0, 255, 0), 2)
        
        # Add instructions on the frame
        cv2.putText(frame, "SPACE: Capture | Q: Quit", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Photos captured: {photo_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Photo Capture - Dual Board Training', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or ESC key
            break
        elif key == ord(' '):  # Spacebar
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # microseconds to milliseconds
            filename = f"photo_{timestamp}.jpg"
            filepath = os.path.join(photos_dir, filename)
            
            # Save the original frame (without the line and text)
            ret, clean_frame = cap.read()
            if ret:
                cv2.imwrite(filepath, clean_frame)
                photo_count += 1
                print(f"Photo saved: {filename}")
            else:
                print("Error: Failed to capture clean frame for saving")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession complete! Captured {photo_count} photos in '{photos_dir}' folder")

if __name__ == "__main__":
    main()
