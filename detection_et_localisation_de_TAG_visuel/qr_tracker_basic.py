#!/usr/bin/env python

"""
QR Code Detection and Tracking - Basic Version

This script detects and tracks QR codes in video streams, similar to the ball tracking
implementation but using QR code detection libraries.

Author: Lucy SAINT-AURET
Date: October 2025
"""

import cv2
import numpy as np
from pyzbar import pyzbar
import time

class QRCodeTracker:
    def __init__(self):
        self.window_name = 'QR Code Tracker'
        self.last_position = (-1, -1)
        self.detection_count = 0
        
    def detect_qr_codes(self, image):
        """
        Detect QR codes in the image using pyzbar library
        Returns list of detected QR codes with their data and positions
        """
        # Convert to grayscale for better detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect QR codes
        qr_codes = pyzbar.decode(gray)
        
        return qr_codes
    
    def draw_qr_code_info(self, image, qr_codes):
        """
        Draw bounding boxes and information for detected QR codes
        """
        for qr_code in qr_codes:
            # Get the bounding box points
            points = qr_code.polygon
            
            # Convert points to numpy array
            if len(points) == 4:
                pts = np.array([[point.x, point.y] for point in points], dtype=np.int32)
                
                # Draw the bounding box
                cv2.polylines(image, [pts], True, (0, 255, 0), 2)
                
                # Calculate center point
                center_x = int(np.mean([point.x for point in points]))
                center_y = int(np.mean([point.y for point in points]))
                
                # Draw center point
                cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), -1)
                
                # Draw QR code data
                qr_data = qr_code.data.decode('utf-8')
                qr_type = qr_code.type
                
                # Display information
                cv2.putText(image, f"Type: {qr_type}", (center_x - 50, center_y - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(image, f"Data: {qr_data[:20]}...", (center_x - 50, center_y - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(image, f"Pos: ({center_x}, {center_y})", (center_x - 50, center_y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                self.last_position = (center_x, center_y)
                self.detection_count += 1
        
        return image
    
    def track_video(self, video_path=None):
        """
        Track QR codes in video file or webcam feed
        """
        # Open video capture
        if video_path:
            cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(0)  # Webcam
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        print("QR Code Tracker started. Press 'q' to quit.")
        print("Press 's' to save current frame with detections")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to read frame")
                break
            
            frame_count += 1
            
            # Detect QR codes
            qr_codes = self.detect_qr_codes(frame)
            
            # Draw detection results
            result_frame = self.draw_qr_code_info(frame.copy(), qr_codes)
            
            # Add frame information
            cv2.putText(result_frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result_frame, f"QR Codes: {len(qr_codes)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result_frame, f"Total detections: {self.detection_count}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow(self.window_name, result_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"qr_detection_frame_{frame_count}.png"
                cv2.imwrite(f"./Images_qrcode/{filename}", result_frame)
                print(f"Saved frame to {filename}")
        
        # Cleanup
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\nTracking completed:")
        print(f"Total frames processed: {frame_count}")
        print(f"Total QR code detections: {self.detection_count}")
        print(f"Average FPS: {fps:.2f}")
        print(f"Last detected position: {self.last_position}")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """
    Main function to run QR code tracking
    """
    tracker = QRCodeTracker()
    
    # You can specify a video file path or use None for webcam
    # video_path = "path/to/your/video.mp4"
    video_path = None  # Use webcam
    
    tracker.track_video(video_path)

if __name__ == "__main__":
    main()