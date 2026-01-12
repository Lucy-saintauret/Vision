#!/usr/bin/env python

"""
Tag Generator - Create QR codes and ArUco markers for testing

This script generates various types of visual tags for testing the detection system.

Author: Lucy SAINT-AURET
Date: October 2025
"""

import cv2
import numpy as np
import qrcode
from PIL import Image
import os

class TagGenerator:
    def __init__(self, output_dir="generated_tags"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def generate_qr_codes(self, data_list=None):
        """Generate QR codes with different data"""
        if data_list is None:
            data_list = [
                "Hello World!",
                "CPE Lyon Vision TP",
                "https://www.cpe.fr",
                "QR Code Test 1",
                "QR Code Test 2",
                "Tag Detection System"
            ]
        
        print("Generating QR codes...")
        for i, data in enumerate(data_list):
            # Create QR code
            qr = qrcode.QRCode(
                version=1,  # Size of the QR code
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(data)
            qr.make(fit=True)
            
            # Create image
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Convert PIL to OpenCV format
            # First convert to RGB mode to ensure proper format
            img_rgb = img.convert('RGB')
            img_cv = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
            
            # Save image
            filename = os.path.join(self.output_dir, f"qr_code_{i+1}.png")
            cv2.imwrite(filename, img_cv)
            print(f"Created: {filename} with data: '{data}'")
    
    def generate_aruco_markers(self, marker_ids=None, dictionary_type=cv2.aruco.DICT_6X6_250):
        """Generate ArUco markers"""
        if marker_ids is None:
            marker_ids = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25]
        
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_type)
        
        print("Generating ArUco markers...")
        for marker_id in marker_ids:
            # Generate marker image
            marker_size = 200  # pixels
            marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)
            
            # Add white border
            border_size = 20
            bordered_img = cv2.copyMakeBorder(marker_img, border_size, border_size, 
                                            border_size, border_size, 
                                            cv2.BORDER_CONSTANT, value=255)
            
            # Save image
            filename = os.path.join(self.output_dir, f"aruco_marker_{marker_id}.png")
            cv2.imwrite(filename, bordered_img)
            print(f"Created: {filename} with ID: {marker_id}")
    
    def create_test_scene(self, scene_name="test_scene"):
        """Create a test scene with multiple tags"""
        # Create a large white canvas
        canvas_width, canvas_height = 800, 600
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
        
        # Load some generated tags and place them on canvas
        qr_files = [f for f in os.listdir(self.output_dir) if f.startswith("qr_code")]
        aruco_files = [f for f in os.listdir(self.output_dir) if f.startswith("aruco_marker")]
        
        positions = [
            (50, 50),     # Top-left
            (400, 50),    # Top-right
            (50, 300),    # Bottom-left
            (400, 300),   # Bottom-right
            (200, 150),   # Center
        ]
        
        tag_files = (qr_files + aruco_files)[:len(positions)]
        
        for i, (tag_file, pos) in enumerate(zip(tag_files, positions)):
            tag_path = os.path.join(self.output_dir, tag_file)
            tag_img = cv2.imread(tag_path)
            
            if tag_img is not None:
                # Resize tag
                tag_size = 150
                tag_resized = cv2.resize(tag_img, (tag_size, tag_size))
                
                # Place on canvas
                x, y = pos
                canvas[y:y+tag_size, x:x+tag_size] = tag_resized
                
                # Add label
                cv2.putText(canvas, f"Tag {i+1}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Save test scene
        scene_filename = os.path.join(self.output_dir, f"{scene_name}.png")
        cv2.imwrite(scene_filename, canvas)
        print(f"Created test scene: {scene_filename}")
        
        return scene_filename
    
    def create_mixed_size_scene(self, scene_name="mixed_size_scene"):
        """Create a scene with tags of different sizes for distance testing"""
        canvas_width, canvas_height = 1000, 700
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
        
        # Generate a few tags specifically for this scene
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        
        # Different sizes to simulate different distances
        sizes = [50, 100, 150, 200, 250]
        positions = [(100, 100), (300, 100), (500, 100), (700, 100), (400, 400)]
        marker_ids = [1, 2, 3, 4, 5]
        
        for size, pos, marker_id in zip(sizes, positions, marker_ids):
            # Generate ArUco marker
            marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, size)
            
            # Add border
            border = size // 10
            bordered = cv2.copyMakeBorder(marker_img, border, border, border, border,
                                        cv2.BORDER_CONSTANT, value=255)
            
            # Place on canvas
            x, y = pos
            h, w = bordered.shape[:2]
            if y + h < canvas_height and x + w < canvas_width:
                canvas[y:y+h, x:x+w] = cv2.cvtColor(bordered, cv2.COLOR_GRAY2BGR)
                
                # Add size label
                cv2.putText(canvas, f"Size: {size}px", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Save scene
        scene_filename = os.path.join(self.output_dir, f"{scene_name}.png")
        cv2.imwrite(scene_filename, canvas)
        print(f"Created mixed size scene: {scene_filename}")
        
        return scene_filename

def main():
    """Generate test tags and scenes"""
    generator = TagGenerator()
    
    print("=== Tag Generator ===")
    
    # Generate QR codes
    generator.generate_qr_codes()
    
    # Generate ArUco markers
    generator.generate_aruco_markers()
    
    # Create test scenes
    generator.create_test_scene()
    generator.create_mixed_size_scene()
    
    print(f"\nAll tags and test scenes generated in '{generator.output_dir}' folder")
    print("You can use these images to test the tag detection system!")

if __name__ == "__main__":
    main()