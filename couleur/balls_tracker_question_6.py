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
        # Threshold the HSV image for only blue colors
        lower_color = np.array([90,70,70])
        upper_color = np.array([130,255,255])
    elif couleur == "rose":
        # Threshold the HSV image for only pink colors
        lower_color = np.array([160, 80, 80])
        upper_color = np.array([179, 255, 255])
    elif couleur == "jaune":
        # Threshold the HSV image for only yellow colors
        lower_color = np.array([20,150,150])
        upper_color = np.array([40,255,255])
    else:
        # Threshold the HSV image for only green colors
        lower_color = np.array([40,70,70])
        upper_color = np.array([80,200,200])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Blur the mask
    bmask = cv2.GaussianBlur(mask, (5,5),0)

    # Alternative method to find centroid
    contours, _ = cv2.findContours(bmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroid_x, centroid_y = None, None
    
    if contours:
        # Trouver le plus gros contour (plus grosse balle)
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter != 0:
            circularity = 4 * math.pi * (area / (perimeter * perimeter))
            if area > 500:  # Ajuste le seuil d'aire si besoin
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius)
                # Dessiner un cercle autour de la balle suivie
                cv2.circle(image, center, radius, (0, 255, 0), 3)
                # Dessiner le centre
                cv2.circle(image, center, 5, (0, 0, 255), -1)
                centroid_x, centroid_y = center

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
    capture = cv2.VideoCapture(0)

    while True:
        okay, image = capture.read()
        if not okay:
            print('Capture failed')
            break

        if not track(image, "rose"):
            break

        if cv2.waitKey(1) & 0xFF == 27:
            break

    capture.release()
    cv2.destroyAllWindows()

