#!/usr/bin/env python

'''
Track a green ball using OpenCV.

    Copyright (C) 2015 Conan Zhao and Simon D. Levy

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as 
    published by the Free Software Foundation, either version 3 of the 
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

 You should have received a copy of the GNU Lesser General Public License 
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import cv2
import math
import numpy as np
from typing import Literal

# For OpenCV2 image display
WINDOW_NAME = 'GreenBallTracker' 

def track(image, couleur: Literal["bleu", "rose", "jaune", "vert"]="vert"):

    '''Accepts BGR image as Numpy array
       Returns: (x,y) coordinates of centroid if found
                (-1,-1) if no centroid was found
                None if user hit ESC
    '''

    # Blur the image to reduce noise
    blur = cv2.GaussianBlur(image, (5,5),0)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    if couleur == "bleu":
        lower_color = np.array([90,70,70])
        upper_color = np.array([130,255,255])
    elif couleur == "rose":
        lower_color = np.array([160, 80, 80])
        upper_color = np.array([179, 255, 255])
    elif couleur == "jaune":
        lower_color = np.array([15,60,60])
        upper_color = np.array([40,255,255])
    else:
        lower_color = np.array([40,70,70])
        upper_color = np.array([80,200,200])

    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Blur the mask
    bmask = cv2.GaussianBlur(mask, (5,5),0)
    contours, _ = cv2.findContours(bmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroid_x, centroid_y = None, None
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * (area / (perimeter * perimeter))
        if area > 10000 and circularity > 0.4:  # Filtrer les petits bruits
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                centroid_x = int(M["m10"] / M["m00"])
                centroid_y = int(M["m01"] / M["m00"])
                if centroid_x != None and centroid_y != None:
                    ctr = (centroid_x, centroid_y)
                    cv2.circle(image, ctr, 10, (0,0,125), -1)

    # Assume no centroid
    ctr = (-1,-1)

    # Use centroid if it exists
    if centroid_x != None and centroid_y != None:

        ctr = (centroid_x, centroid_y)

    # Display full-color image
    cv2.imshow(WINDOW_NAME, image)

    # Force image display, setting centroid to None on ESC key input
    if cv2.waitKey(1) & 0xFF == 27:
        ctr = None
    
    # Return coordinates of centroid
    return ctr

# Test with input from camera
if __name__ == '__main__':
    
    capture = cv2.VideoCapture('ball3.mp4')

    while True:

        okay, image = capture.read()

        if okay:

            if not track(image, "bleu"):
                break
          
            if cv2.waitKey(1) & 0xFF == 27:
                break

        else:

           print('Capture failed')
           break

