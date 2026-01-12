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
last_center = None

def track(image, couleur: Literal["bleu", "rose", "jaune", "vert"]="vert", last_center=None):

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

    # Take the moments to get the centroid
    # moments = cv2.moments(bmask)
    # m00 = moments['m00']
    # centroid_x, centroid_y = None, None
    # if m00 != 0:
    #     centroid_x = int(moments['m10']/m00)
    #     centroid_y = int(moments['m01']/m00)

    # Alternative method to find centroid
    contours, _ = cv2.findContours(bmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroid_x, centroid_y = None, None

    if contours:
        centers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                centers.append(((int(x), int(y)), int(radius), cnt))
        if centers:
            if last_center is None:
                # Prendre la plus grosse au début
                center, radius, cnt = max(centers, key=lambda c: c[1])
            else:
                # Prendre la plus proche du dernier centre
                center, radius, cnt = min(
                    centers, key=lambda c: (c[0][0] - last_center[0])**2 + (c[0][1] - last_center[1])**2
                )
            cv2.circle(image, center, radius, (0, 255, 0), 3)
            cv2.circle(image, center, 5, (0, 0, 255), -1)
            centroid_x, centroid_y = center

    ctr = (-1, -1)
    if centroid_x is not None and centroid_y is not None:
        ctr = (centroid_x, centroid_y)

    cv2.imshow(WINDOW_NAME, image)
    if cv2.waitKey(1) & 0xFF == 27:
        ctr = None
    return ctr

# Test with input from camera
if __name__ == '__main__':
    capture = cv2.VideoCapture(0)
    last_center = None

    while True:
        okay, image = capture.read()
        if not okay:
            print('Capture failed')
            break

        result = track(image, "rose", last_center)
        if result is None:
            break
        if result != (-1, -1):
            last_center = result

    capture.release()
    cv2.destroyAllWindows()

