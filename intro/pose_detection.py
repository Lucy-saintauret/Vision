#!/usr/bin/env python3
"""
Détection de Poses Corporelles avec MediaPipe
============================================

Ce script utilise MediaPipe pour détecter les poses corporelles en temps réel.
Il peut détecter 33 points du corps et analyser les postures basiques.

Poses analysées :
- Debout
- Bras levés
- Bras croisés
- Penché en avant
- Assis (approximatif)

Utilisation : python pose_detection.py
Appuyez sur 'q' pour quitter.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def calculate_angle(self, a, b, c):
        """Calcule l'angle entre trois points"""
        a = np.array(a)
        b = np.array(b)  # Point central
        c = np.array(c)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def get_landmark_coords(self, landmarks, landmark_id):
        """Récupère les coordonnées d'un landmark"""
        return [landmarks[landmark_id].x, landmarks[landmark_id].y]
    
    def analyze_pose(self, landmarks):
        """Analyse la pose et retourne une description"""
        try:
            # Points clés pour l'analyse
            left_shoulder = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER)
            right_shoulder = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
            left_elbow = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_ELBOW)
            right_elbow = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_ELBOW)
            left_wrist = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_WRIST)
            right_wrist = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_WRIST)
            left_hip = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP)
            right_hip = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_HIP)
            left_knee = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_KNEE)
            right_knee = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_KNEE)
            nose = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.NOSE)
            
            # Calculer des angles
            left_arm_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_arm_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Position des bras
            left_arm_up = left_wrist[1] < left_shoulder[1]
            right_arm_up = right_wrist[1] < right_shoulder[1]
            
            # Posture du tronc
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_center_y = (left_hip[1] + right_hip[1]) / 2
            
            # Analyse de la posture
            poses = []
            
            # Bras levés
            if left_arm_up and right_arm_up:
                poses.append("Bras leves")
            elif left_arm_up:
                poses.append("Bras gauche leve")
            elif right_arm_up:
                poses.append("Bras droit leve")
            
            # Bras croisés (approximation)
            if (left_wrist[0] > right_shoulder[0] and right_wrist[0] < left_shoulder[0] and
                not left_arm_up and not right_arm_up):
                poses.append("Bras croises")
            
            # Position debout/assis (approximation basée sur la position des genoux par rapport aux hanches)
            knee_hip_ratio = (left_knee[1] + right_knee[1]) / (left_hip[1] + right_hip[1])
            if knee_hip_ratio > 1.3:
                poses.append("Assis")
            else:
                poses.append("Debout")
            
            # Penché en avant (tête plus basse que les épaules)
            if nose[1] > shoulder_center_y:
                poses.append("Penche en avant")
            
            return " | ".join(poses) if poses else "Pose inconnue"
            
        except Exception as e:
            return f"Erreur d'analyse: {str(e)}"
    
    def process_frame(self, frame):
        """Traite une frame et retourne l'image avec annotations"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Dessiner les landmarks
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            # Analyser la pose
            pose_description = self.analyze_pose(results.pose_landmarks.landmark)
            
            # Afficher la description de la pose
            y_pos = 50
            for i, line in enumerate(pose_description.split(" | ")):
                cv2.putText(frame, line, (10, y_pos + i * 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Afficher quelques angles pour debug
            try:
                landmarks = results.pose_landmarks.landmark
                left_shoulder = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER)
                left_elbow = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_ELBOW)
                left_wrist = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_WRIST)
                
                left_arm_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
                cv2.putText(frame, f"Angle bras gauche: {int(left_arm_angle)}°", 
                           (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            except:
                pass
        
        return frame

def main():
    print("Démarrage de la détection de poses...")
    print("Poses détectées : Debout/Assis, Bras levés, Bras croisés, Penché en avant")
    print("Appuyez sur 'q' pour quitter")
    
    detector = PoseDetector()
    cap = cv2.VideoCapture(0)
    
    # Vérifier que la caméra est bien ouverte
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la webcam")
        return
    
    # Variables pour calculer les FPS
    fps_counter = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur : impossible de lire la frame")
            break
        
        # Retourner l'image horizontalement pour un effet miroir
        frame = cv2.flip(frame, 1)
        
        # Traiter la frame
        frame = detector.process_frame(frame)
        
        # Calculer et afficher les FPS
        fps_counter += 1
        if fps_counter % 30 == 0:
            end_time = time.time()
            fps = 30 / (end_time - start_time)
            start_time = end_time
            print(f"FPS: {fps:.1f}")
        
        # Afficher les instructions
        cv2.putText(frame, "Essayez: lever les bras, croiser les bras, vous pencher", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Afficher la frame
        cv2.imshow('Detection de Poses', frame)
        
        # Quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Application fermée")

if __name__ == "__main__":
    main()