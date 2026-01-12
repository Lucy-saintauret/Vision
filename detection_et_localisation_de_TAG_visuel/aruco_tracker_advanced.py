#!/usr/bin/env python

"""
ArUco Marker Detection and Tracking - Advanced Version

This script uses OpenCV's ArUco marker detection for more robust tracking
and pose estimation capabilities.

Author: Lucy SAINT-AURET
Date: October 2025
"""

import cv2
import numpy as np
import time

class ArUcoTracker:
    def __init__(self, dictionary_type=cv2.aruco.DICT_6X6_250):
        self.window_name = 'ArUco Marker Tracker'
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_type)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        
        # Tracking variables
        self.last_positions = {}  # Store positions for each marker ID
        self.detection_count = 0
        self.marker_sizes = {}  # Store size of each marker for distance estimation
        
        # Camera calibration parameters (you should calibrate your camera)
        # These are example values - replace with actual calibration data
        self.camera_matrix = np.array([[800, 0, 320],
                                     [0, 800, 240],
                                     [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.array([0.1, -0.2, 0, 0, 0], dtype=np.float32)
        self.marker_length = 0.05  # 5cm marker size in meters
        
    def detect_aruco_markers(self, image):
        """
        Detect ArUco markers in the image
        Returns corners, ids, and rejected candidates
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        return corners, ids, rejected
    
    def estimate_pose(self, corners):
        """
        Estimate pose (rotation and translation) of markers
        """
        if len(corners) > 0:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs)
            return rvecs, tvecs
        return None, None
    
    def calculate_distance(self, tvec):
        """
        Calculate distance from camera to marker using translation vector
        """
        if tvec is not None:
            return np.linalg.norm(tvec)
        return 0
    
    def calculate_marker_area(self, corners):
        """
        Calculate the area of the marker for size-based distance estimation
        """
        if corners is not None and len(corners) > 0:
            corner_points = corners[0]
            # Calculate area using cross product
            v1 = corner_points[1] - corner_points[0]
            v2 = corner_points[3] - corner_points[0]
            area = abs(np.cross(v1, v2))
            return area
        return 0
    
    def draw_marker_info(self, image, corners, ids, rvecs=None, tvecs=None):
        """
        Draw detection results and pose information
        """
        if ids is not None:
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
            
            for i, marker_id in enumerate(ids.flatten()):
                # Get corner points
                corner_points = corners[i][0]
                
                # Calculate center
                center_x = int(np.mean(corner_points[:, 0]))
                center_y = int(np.mean(corner_points[:, 1]))
                
                # Store position
                self.last_positions[marker_id] = (center_x, center_y)
                
                # Calculate marker area for distance estimation
                area = self.calculate_marker_area(corners[i])
                self.marker_sizes[marker_id] = area
                
                # Estimate distance based on area (empirical method)
                if area > 0:
                    # Empirical formula: distance inversely proportional to sqrt(area)
                    estimated_distance = 10000 / np.sqrt(area)  # Rough estimate
                else:
                    estimated_distance = 0
                
                # Draw center point
                cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), -1)
                
                # Draw marker information
                info_y = center_y - 60
                cv2.putText(image, f"ID: {marker_id}", (center_x - 50, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(image, f"Pos: ({center_x}, {center_y})", (center_x - 50, info_y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(image, f"Area: {area:.0f}", (center_x - 50, info_y + 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(image, f"Dist: {estimated_distance:.1f}cm", (center_x - 50, info_y + 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Draw pose axes if pose estimation is available
                if rvecs is not None and tvecs is not None:
                    cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, 
                                    rvecs[i], tvecs[i], self.marker_length)
                    
                    # Calculate and display 3D distance
                    distance_3d = self.calculate_distance(tvecs[i])
                    cv2.putText(image, f"3D Dist: {distance_3d*100:.1f}cm", 
                               (center_x - 50, info_y + 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                
                self.detection_count += 1
        
        return image
    
    def track_video(self, video_path=None, enable_pose_estimation=True):
        """
        Track ArUco markers in video file or webcam feed
        """
        # Open video capture
        if video_path:
            cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        print("ArUco Marker Tracker started. Press 'q' to quit.")
        print("Press 's' to save current frame with detections")
        print("Press 'p' to toggle pose estimation")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to read frame")
                break
            
            frame_count += 1
            
            # Detect ArUco markers
            corners, ids, rejected = self.detect_aruco_markers(frame)
            
            # Estimate pose if enabled
            rvecs, tvecs = None, None
            if enable_pose_estimation and corners:
                rvecs, tvecs = self.estimate_pose(corners)
            
            # Draw detection results
            result_frame = self.draw_marker_info(frame.copy(), corners, ids, rvecs, tvecs)
            
            # Add frame information
            num_markers = len(ids) if ids is not None else 0
            cv2.putText(result_frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result_frame, f"Markers: {num_markers}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result_frame, f"Total detections: {self.detection_count}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result_frame, f"Pose est: {'ON' if enable_pose_estimation else 'OFF'}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display the frame
            cv2.imshow(self.window_name, result_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"aruco_detection_frame_{frame_count}.png"
                cv2.imwrite(f"./Images_tags/{filename}", result_frame)
                print(f"Saved frame to {filename}")
            elif key == ord('p'):
                enable_pose_estimation = not enable_pose_estimation
                print(f"Pose estimation {'enabled' if enable_pose_estimation else 'disabled'}")
        
        # Cleanup and statistics
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\nTracking completed:")
        print(f"Total frames processed: {frame_count}")
        print(f"Total marker detections: {self.detection_count}")
        print(f"Unique markers detected: {len(self.last_positions)}")
        print(f"Average FPS: {fps:.2f}")
        print(f"Last positions: {self.last_positions}")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """
    Main function to run ArUco marker tracking
    """
    # Available dictionary types:
    # cv2.aruco.DICT_4X4_50, DICT_4X4_100, DICT_4X4_250, DICT_4X4_1000
    # cv2.aruco.DICT_5X5_50, DICT_5X5_100, DICT_5X5_250, DICT_5X5_1000
    # cv2.aruco.DICT_6X6_50, DICT_6X6_100, DICT_6X6_250, DICT_6X6_1000
    # cv2.aruco.DICT_7X7_50, DICT_7X7_100, DICT_7X7_250, DICT_7X7_1000
    
    tracker = ArUcoTracker(cv2.aruco.DICT_6X6_250)
    
    # You can specify a video file path or use None for webcam
    video_path = None  # Use webcam
    
    tracker.track_video(video_path, enable_pose_estimation=True)

if __name__ == "__main__":
    main()