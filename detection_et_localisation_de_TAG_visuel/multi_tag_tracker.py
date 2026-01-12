#!/usr/bin/env python

"""
Multi-Tag Tracker - Combined QR Code and ArUco Detection

This script combines both QR code and ArUco marker detection for comprehensive
tag tracking, similar to multi-ball tracking but for visual tags.

Author: Lucy SAINT-AURET
Date: October 2025
"""

import cv2
import numpy as np
from pyzbar import pyzbar
import time

class MultiTagTracker:
    def __init__(self, aruco_dict_type=cv2.aruco.DICT_6X6_250):
        self.window_name = 'Multi-Tag Tracker'
        
        # ArUco setup
        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        self.aruco_parameters = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dictionary, self.aruco_parameters)
        
        # Tracking variables
        self.qr_positions = []
        self.aruco_positions = {}
        self.total_detections = 0
        self.tracking_history = []  # For continuity tracking
        
        # Colors for different tag types
        self.qr_color = (0, 255, 0)      # Green for QR codes
        self.aruco_color = (255, 0, 0)   # Blue for ArUco
        self.center_color = (0, 0, 255)  # Red for centers
        
    def detect_qr_codes(self, image):
        """Detect QR codes using pyzbar"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        qr_codes = pyzbar.decode(gray)
        
        qr_data = []
        for qr in qr_codes:
            # Calculate center and bounding box
            points = qr.polygon
            if len(points) == 4:
                center_x = int(np.mean([p.x for p in points]))
                center_y = int(np.mean([p.y for p in points]))
                
                # Calculate bounding box
                x_coords = [p.x for p in points]
                y_coords = [p.y for p in points]
                bbox = (min(x_coords), min(y_coords), 
                       max(x_coords) - min(x_coords), 
                       max(y_coords) - min(y_coords))
                
                qr_data.append({
                    'type': 'QR',
                    'center': (center_x, center_y),
                    'bbox': bbox,
                    'data': qr.data.decode('utf-8'),
                    'polygon': points,
                    'area': bbox[2] * bbox[3]
                })
        
        return qr_data
    
    def detect_aruco_markers(self, image):
        """Detect ArUco markers"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        
        aruco_data = []
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                corner_points = corners[i][0]
                
                # Calculate center
                center_x = int(np.mean(corner_points[:, 0]))
                center_y = int(np.mean(corner_points[:, 1]))
                
                # Calculate bounding box
                x_coords = corner_points[:, 0]
                y_coords = corner_points[:, 1]
                bbox = (int(min(x_coords)), int(min(y_coords)), 
                       int(max(x_coords) - min(x_coords)), 
                       int(max(y_coords) - min(y_coords)))
                
                # Calculate area
                area = bbox[2] * bbox[3]
                
                aruco_data.append({
                    'type': 'ArUco',
                    'id': marker_id,
                    'center': (center_x, center_y),
                    'bbox': bbox,
                    'corners': corner_points,
                    'area': area
                })
        
        return aruco_data
    
    def track_largest_tag(self, all_tags):
        """
        Track the largest tag (similar to largest ball tracking)
        """
        if not all_tags:
            return None
        
        # Find the tag with largest area
        largest_tag = max(all_tags, key=lambda x: x['area'])
        return largest_tag
    
    def track_closest_to_previous(self, all_tags, previous_position, max_distance=100):
        """
        Track the tag closest to previous position (continuity tracking)
        """
        if not all_tags or previous_position == (-1, -1):
            return self.track_largest_tag(all_tags)
        
        min_distance = float('inf')
        closest_tag = None
        
        for tag in all_tags:
            distance = np.sqrt((tag['center'][0] - previous_position[0])**2 + 
                             (tag['center'][1] - previous_position[1])**2)
            
            if distance < min_distance and distance < max_distance:
                min_distance = distance
                closest_tag = tag
        
        # If no tag is close enough, select the largest
        return closest_tag if closest_tag else self.track_largest_tag(all_tags)
    
    def estimate_distance_from_area(self, area, tag_type):
        """
        Estimate distance based on tag area (empirical method)
        """
        if area <= 0:
            return 0
        
        # Different scaling factors for different tag types
        if tag_type == 'QR':
            # QR codes are typically larger
            scale_factor = 15000
        else:  # ArUco
            scale_factor = 12000
        
        estimated_distance = scale_factor / np.sqrt(area)
        return estimated_distance
    
    def draw_tag_info(self, image, tag, color):
        """Draw information for a single tag"""
        center = tag['center']
        bbox = tag['bbox']
        
        # Draw bounding box
        cv2.rectangle(image, (bbox[0], bbox[1]), 
                     (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 2)
        
        # Draw center point
        cv2.circle(image, center, 5, self.center_color, -1)
        
        # Estimate distance
        distance = self.estimate_distance_from_area(tag['area'], tag['type'])
        
        # Prepare text information
        info_lines = [
            f"Type: {tag['type']}",
            f"Pos: {center}",
            f"Area: {tag['area']:.0f}",
            f"Dist: {distance:.1f}cm"
        ]
        
        # Add specific information based on tag type
        if tag['type'] == 'QR':
            info_lines.append(f"Data: {tag['data'][:15]}...")
        elif tag['type'] == 'ArUco':
            info_lines.append(f"ID: {tag['id']}")
        
        # Draw text information
        for i, line in enumerate(info_lines):
            y_pos = center[1] - 80 + i * 20
            cv2.putText(image, line, (center[0] - 70, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return image
    
    def track_video(self, video_path=None, tracking_mode='largest'):
        """
        Track tags in video with different tracking strategies
        tracking_mode: 'largest', 'continuity', 'all'
        """
        # Open video capture
        if video_path:
            cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        print(f"Multi-Tag Tracker started with '{tracking_mode}' mode.")
        print("Press 'q' to quit, 's' to save frame")
        print("Press '1' for largest tracking, '2' for continuity tracking, '3' for all tags")
        
        frame_count = 0
        start_time = time.time()
        last_tracked_position = (-1, -1)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to read frame")
                break
            
            frame_count += 1
            
            # Detect all types of tags
            qr_tags = self.detect_qr_codes(frame)
            aruco_tags = self.detect_aruco_markers(frame)
            all_tags = qr_tags + aruco_tags
            
            result_frame = frame.copy()
            
            # Apply tracking strategy
            if tracking_mode == 'largest':
                tracked_tag = self.track_largest_tag(all_tags)
                if tracked_tag:
                    self.draw_tag_info(result_frame, tracked_tag, (0, 255, 255))  # Yellow for tracked
                    last_tracked_position = tracked_tag['center']
                    
            elif tracking_mode == 'continuity':
                tracked_tag = self.track_closest_to_previous(all_tags, last_tracked_position)
                if tracked_tag:
                    self.draw_tag_info(result_frame, tracked_tag, (0, 255, 255))  # Yellow for tracked
                    last_tracked_position = tracked_tag['center']
                    
            elif tracking_mode == 'all':
                # Draw all detected tags
                for tag in qr_tags:
                    self.draw_tag_info(result_frame, tag, self.qr_color)
                for tag in aruco_tags:
                    self.draw_tag_info(result_frame, tag, self.aruco_color)
            
            # Update counters
            self.total_detections += len(all_tags)
            
            # Add frame information
            cv2.putText(result_frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result_frame, f"Mode: {tracking_mode}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result_frame, f"QR: {len(qr_tags)}, ArUco: {len(aruco_tags)}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result_frame, f"Total detections: {self.total_detections}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display the frame
            cv2.imshow(self.window_name, result_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"multi_tag_detection_frame_{frame_count}.png"
                cv2.imwrite(f"./Images_mult/{filename}", result_frame)
                print(f"Saved frame to {filename}")
            elif key == ord('1'):
                tracking_mode = 'largest'
                print("Switched to largest tag tracking")
            elif key == ord('2'):
                tracking_mode = 'continuity'
                print("Switched to continuity tracking")
            elif key == ord('3'):
                tracking_mode = 'all'
                print("Switched to all tags display")
        
        # Cleanup and statistics
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\nTracking completed:")
        print(f"Total frames processed: {frame_count}")
        print(f"Total tag detections: {self.total_detections}")
        print(f"Average FPS: {fps:.2f}")
        print(f"Final tracking mode: {tracking_mode}")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """
    Main function to run multi-tag tracking
    """
    tracker = MultiTagTracker()
    
    # Start with webcam and 'all' mode to see everything
    tracker.track_video(video_path=None, tracking_mode='all')

if __name__ == "__main__":
    main()