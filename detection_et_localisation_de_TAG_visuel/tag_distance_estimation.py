#!/usr/bin/env python

"""
Tag Distance Estimation and Calibration

This script implements distance estimation for visual tags using both
empirical methods and camera calibration, similar to the ball distance estimation.

Author: Lucy SAINT-AURET
Date: October 2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pyzbar import pyzbar
import json
import os

class TagDistanceEstimator:
    def __init__(self):
        self.calibration_data = {}
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibrated = False
        
        # ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Real-world tag size (in meters)
        self.real_aruco_size = 0.05  # 5cm
        self.real_qr_size = 0.03     # 3cm (approximate)
    
    def collect_calibration_data(self, video_path=None):
        """
        Collect calibration data by asking user to position tags at known distances
        """
        print("=== Tag Distance Calibration Data Collection ===")
        print("Position a tag at different known distances and press 's' to save data")
        print("Press 'q' to quit and process collected data")
        
        cap = cv2.VideoCapture(video_path if video_path else 0)
        calibration_points = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect tags
            qr_tags = self.detect_qr_codes(frame)
            aruco_corners, aruco_ids, _ = self.aruco_detector.detectMarkers(frame)
            
            display_frame = frame.copy()
            
            # Draw QR codes
            for qr in qr_tags:
                points = qr.polygon
                if len(points) == 4:
                    pts = np.array([[p.x, p.y] for p in points], dtype=np.int32)
                    cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)
                    
                    # Calculate area
                    area = cv2.contourArea(pts)
                    center_x = int(np.mean([p.x for p in points]))
                    center_y = int(np.mean([p.y for p in points]))
                    
                    cv2.putText(display_frame, f"QR Area: {area:.0f}", 
                               (center_x-50, center_y-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw ArUco markers
            if aruco_ids is not None:
                cv2.aruco.drawDetectedMarkers(display_frame, aruco_corners, aruco_ids)
                for i, marker_id in enumerate(aruco_ids.flatten()):
                    corners = aruco_corners[i][0]
                    area = cv2.contourArea(corners)
                    center_x = int(np.mean(corners[:, 0]))
                    center_y = int(np.mean(corners[:, 1]))
                    
                    cv2.putText(display_frame, f"ArUco Area: {area:.0f}", 
                               (center_x-50, center_y+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Display instructions
            cv2.putText(display_frame, "Position tag at known distance, press 's' to save", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Collected points: {len(calibration_points)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Distance Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Ask for distance
                print("\nEnter the actual distance to the tag (in cm): ", end="")
                try:
                    distance_cm = float(input())
                    
                    # Save data for QR codes
                    for qr in qr_tags:
                        points = qr.polygon
                        if len(points) == 4:
                            pts = np.array([[p.x, p.y] for p in points], dtype=np.int32)
                            area = cv2.contourArea(pts)
                            calibration_points.append({
                                'type': 'QR',
                                'distance_cm': distance_cm,
                                'area': area
                            })
                            print(f"Saved QR calibration point: {distance_cm}cm, area: {area}")
                    
                    # Save data for ArUco markers
                    if aruco_ids is not None:
                        for i, marker_id in enumerate(aruco_ids.flatten()):
                            corners = aruco_corners[i][0]
                            area = cv2.contourArea(corners)
                            calibration_points.append({
                                'type': 'ArUco',
                                'id': marker_id,
                                'distance_cm': distance_cm,
                                'area': area
                            })
                            print(f"Saved ArUco calibration point: {distance_cm}cm, area: {area}")
                            
                except ValueError:
                    print("Invalid distance value")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Process calibration data
        if calibration_points:
            self.process_calibration_data(calibration_points)
        
        return calibration_points
    
    def detect_qr_codes(self, image):
        """Detect QR codes using pyzbar"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return pyzbar.decode(gray)
    
    def process_calibration_data(self, calibration_points):
        """
        Process calibration data to create distance estimation models
        """
        if not calibration_points:
            print("No calibration data available")
            return
        
        # Separate QR and ArUco data
        qr_data = [p for p in calibration_points if p['type'] == 'QR']
        aruco_data = [p for p in calibration_points if p['type'] == 'ArUco']
        
        # Fit models for each type
        models = {}
        
        if qr_data:
            distances = [p['distance_cm'] for p in qr_data]
            areas = [p['area'] for p in qr_data]
            models['QR'] = self.fit_distance_model(areas, distances, 'QR Codes')
        
        if aruco_data:
            distances = [p['distance_cm'] for p in aruco_data]
            areas = [p['area'] for p in aruco_data]
            models['ArUco'] = self.fit_distance_model(areas, distances, 'ArUco Markers')
        
        self.calibration_data = models
        
        # Save calibration data
        self.save_calibration_data(calibration_points, models)
    
    def fit_distance_model(self, areas, distances, tag_type):
        """
        Fit a model: distance = a / sqrt(area) + b
        """
        if len(areas) < 2:
            print(f"Not enough data points for {tag_type}")
            return None
        
        # Convert to numpy arrays
        areas = np.array(areas)
        distances = np.array(distances)
        
        # Create feature matrix: [1/sqrt(area), 1]
        X = np.column_stack([1/np.sqrt(areas), np.ones(len(areas))])
        
        # Fit linear model
        coeffs, residuals, rank, s = np.linalg.lstsq(X, distances, rcond=None)
        a, b = coeffs
        
        print(f"\n{tag_type} distance model: distance = {a:.2f} / sqrt(area) + {b:.2f}")
        
        # Calculate R-squared
        y_pred = a / np.sqrt(areas) + b
        ss_res = np.sum((distances - y_pred) ** 2)
        ss_tot = np.sum((distances - np.mean(distances)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        print(f"R-squared: {r_squared:.3f}")
        
        # Plot calibration curve
        self.plot_calibration_curve(areas, distances, a, b, tag_type)
        
        return {'a': a, 'b': b, 'r_squared': r_squared}
    
    def plot_calibration_curve(self, areas, distances, a, b, tag_type):
        """Plot the calibration curve"""
        plt.figure(figsize=(10, 6))
        
        # Plot data points
        plt.scatter(areas, distances, color='blue', alpha=0.7, label='Measured data')
        
        # Plot fitted curve
        area_range = np.linspace(min(areas), max(areas), 100)
        distance_pred = a / np.sqrt(area_range) + b
        plt.plot(area_range, distance_pred, 'r-', label=f'Fitted curve: d = {a:.2f}/√area + {b:.2f}')
        
        plt.xlabel('Tag Area (pixels²)')
        plt.ylabel('Distance (cm)')
        plt.title(f'{tag_type} Distance Calibration')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(f'tag_distance_calibration_{tag_type.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def estimate_distance(self, area, tag_type):
        """
        Estimate distance using calibrated model
        """
        if tag_type not in self.calibration_data or self.calibration_data[tag_type] is None:
            # Use default empirical formula
            if tag_type == 'QR':
                return 12000 / np.sqrt(area) if area > 0 else 0
            else:  # ArUco
                return 10000 / np.sqrt(area) if area > 0 else 0
        
        model = self.calibration_data[tag_type]
        a, b = model['a'], model['b']
        return a / np.sqrt(area) + b if area > 0 else 0
    
    def save_calibration_data(self, calibration_points, models):
        """Save calibration data to file"""
        data = {
            'calibration_points': calibration_points,
            'models': models,
            'timestamp': str(np.datetime64('now'))
        }
        
        with open('tag_distance_calibration.json', 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print("Calibration data saved to 'tag_distance_calibration.json'")
    
    def load_calibration_data(self, filename='tag_distance_calibration.json'):
        """Load calibration data from file"""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
                self.calibration_data = data.get('models', {})
                print(f"Loaded calibration data from {filename}")
                return True
        return False
    
    def test_distance_estimation(self, video_path=None):
        """
        Test the distance estimation system
        """
        cap = cv2.VideoCapture(video_path if video_path else 0)
        
        print("=== Distance Estimation Test ===")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            display_frame = frame.copy()
            
            # Detect QR codes
            qr_codes = self.detect_qr_codes(frame)
            for qr in qr_codes:
                points = qr.polygon
                if len(points) == 4:
                    pts = np.array([[p.x, p.y] for p in points], dtype=np.int32)
                    area = cv2.contourArea(pts)
                    distance = self.estimate_distance(area, 'QR')
                    
                    # Draw detection
                    cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)
                    center_x = int(np.mean([p.x for p in points]))
                    center_y = int(np.mean([p.y for p in points]))
                    
                    cv2.putText(display_frame, f"QR: {distance:.1f}cm", 
                               (center_x-50, center_y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Detect ArUco markers
            corners, ids, _ = self.aruco_detector.detectMarkers(frame)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)
                for i, marker_id in enumerate(ids.flatten()):
                    corner_points = corners[i][0]
                    area = cv2.contourArea(corner_points)
                    distance = self.estimate_distance(area, 'ArUco')
                    
                    center_x = int(np.mean(corner_points[:, 0]))
                    center_y = int(np.mean(corner_points[:, 1]))
                    
                    cv2.putText(display_frame, f"ArUco {marker_id}: {distance:.1f}cm", 
                               (center_x-50, center_y+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            cv2.imshow('Distance Estimation Test', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """
    Main function for distance estimation
    """
    estimator = TagDistanceEstimator()
    
    print("=== Tag Distance Estimation System ===")
    print("1. Collect calibration data")
    print("2. Load existing calibration")
    print("3. Test distance estimation")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            estimator.collect_calibration_data()
        elif choice == '2':
            if estimator.load_calibration_data():
                print("Calibration data loaded successfully")
            else:
                print("No calibration file found")
        elif choice == '3':
            estimator.test_distance_estimation()
        elif choice == '4':
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()