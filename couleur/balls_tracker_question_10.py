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

def track(image, lower_color, upper_color, last_center=None):

    '''Accepts BGR image as Numpy array
       Returns: (x,y) coordinates of centroid if found
                (-1,-1) if no centroid was found
                None if user hit ESC
    '''

    # Blur the image to reduce noise
    blur = cv2.GaussianBlur(image, (5,5),0)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Utilise directement les seuils passés en argument
    mask = cv2.inRange(hsv, lower_color, upper_color)
    bmask = cv2.GaussianBlur(mask, (5,5),0)

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

def learn_color(image):
    h, w, _ = image.shape
    cx, cy = w // 2, h // 2
    size = 20  # taille du carré central
    roi = image[cy-size//2:cy+size//2, cx-size//2:cx+size//2]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mean = hsv_roi.mean(axis=(0,1))
    return mean, hsv_roi  # (H, S, V)

# Test with input from camera
if __name__ == '__main__':
    capture = cv2.VideoCapture(0)
    last_center = None

    # Phase d'apprentissage
    print("Place l'objet au centre et appuie sur ESPACE pour apprendre la couleur.")
    while True:
        okay, image = capture.read()
        if not okay:
            print('Capture failed')
            break
        h, w, _ = image.shape
        cx, cy = w // 2, h // 2
        size = 20
        # Dessine un carré au centre pour guider l'utilisateur
        cv2.rectangle(image, (cx-size//2, cy-size//2), (cx+size//2, cy+size//2), (255,0,0), 2)
        cv2.imshow(WINDOW_NAME, image)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # ESPACE
            mean_hsv, hsv_roi = learn_color(image)
            print("HSV appris :", mean_hsv)
            break
        elif key == 27:  # ESC
            capture.release()
            cv2.destroyAllWindows()
            exit()

    # Définir les seuils HSV autour de la couleur apprise
    h, s, v = mean_hsv
    h_std, s_std, v_std = np.std(hsv_roi, axis=(0,1))
    k = 1.5 # facteur d'élargissement
    lower_color = np.array([max(0, int(h - k*h_std)), 
                            max(0, int(s - k*s_std)), 
                            max(0, int(v - k*v_std))], 
                           dtype=np.uint8)
    upper_color = np.array([min(179, int(h + k*h_std)), 
                            min(255, int(s + k*s_std)), 
                            min(255, int(v + k*v_std))], 
                           dtype=np.uint8)


    # Boucle de suivi
    while True:
        okay, image = capture.read()
        if not okay:
            print('Capture failed')
            break

        result = track(image, lower_color, upper_color, last_center)
        if result is None:
            break
        if result != (-1, -1):
            last_center = result

    capture.release()
    cv2.destroyAllWindows()

