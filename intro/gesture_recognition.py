#!/usr/bin/env python3
"""
Reconnaissance de Gestes avec MediaPipe
=======================================

Ce script utilise MediaPipe pour détecter et reconnaître des gestes de la main en temps réel.
Il peut reconnaître plusieurs gestes basiques comme le pouce levé, la paume ouverte, le poing, etc.

Gestes reconnus :
- Paume ouverte (tous les doigts étendus)
- Poing fermé (tous les doigts pliés)
- Pouce levé (thumbs up)
- Pouce vers le bas (thumbs down)
- Victoire (index et majeur levés)
- OK (pouce et index en cercle)

Utilisation : python gesture_recognition.py
Appuyez sur 'q' pour quitter.
"""

import cv2
import mediapipe as mp
import numpy as np
import time

class GestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def get_finger_states(self, landmarks):
        """Détermine l'état de chaque doigt (levé ou baissé)"""
        # Points des extrémités des doigts et articulations
        finger_tips = [4, 8, 12, 16, 20]  # Pouce, Index, Majeur, Annulaire, Auriculaire
        finger_pips = [3, 6, 10, 14, 18]  # Articulations précédentes
        
        fingers = []
        
        # Pouce (logique différente car orientation différente)
        if landmarks[finger_tips[0]].x > landmarks[finger_pips[0]].x:
            fingers.append(1)
        else:
            fingers.append(0)
            
        # Autres doigts
        for i in range(1, 5):
            if landmarks[finger_tips[i]].y < landmarks[finger_pips[i]].y:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers
    
    def recognize_gesture(self, finger_states):
        """Reconnaît le geste basé sur l'état des doigts"""
        fingers = finger_states
        
        # Paume ouverte - tous les doigts levés
        if fingers == [1, 1, 1, 1, 1]:
            return "Paume ouverte"
        
        # Poing fermé - tous les doigts baissés
        elif fingers == [0, 0, 0, 0, 0]:
            return "Poing ferme"
        
        # Pouce levé
        elif fingers == [1, 0, 0, 0, 0]:
            return "Pouce leve"
        
        # Pouce vers le bas
        elif fingers == [0, 0, 0, 0, 0]:  # On vérifie aussi l'orientation
            return "Pouce bas"
        
        # Victoire (V) - index et majeur levés
        elif fingers == [0, 1, 1, 0, 0]:
            return "Victoire (V)"
        
        # OK - pouce et index en cercle, autres doigts levés
        elif fingers == [1, 0, 1, 1, 1]:
            return "OK"
        
        # Pointage - index levé seulement
        elif fingers == [0, 1, 0, 0, 0]:
            return "Pointage"
        
        # Rock and roll - index et auriculaire levés
        elif fingers == [0, 1, 0, 0, 1]:
            return "Rock and Roll"
        
        else:
            return "Geste inconnu"
    
    def process_frame(self, frame):
        """Traite une frame et retourne l'image avec annotations"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dessiner les landmarks
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Reconnaître le geste
                finger_states = self.get_finger_states(hand_landmarks.landmark)
                gesture = self.recognize_gesture(finger_states)
                
                # Afficher le geste détecté
                cv2.putText(frame, gesture, (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Afficher l'état des doigts pour debug
                fingers_text = f"Doigts: {finger_states}"
                cv2.putText(frame, fingers_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return frame

def main():
    print("Démarrage de la reconnaissance de gestes...")
    print("Gestes reconnus : Paume ouverte, Poing fermé, Pouce levé, Victoire, OK, Pointage, Rock and Roll")
    print("Appuyez sur 'q' pour quitter")
    
    recognizer = GestureRecognizer()
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
        frame = recognizer.process_frame(frame)
        
        # Calculer et afficher les FPS
        fps_counter += 1
        if fps_counter % 30 == 0:
            end_time = time.time()
            fps = 30 / (end_time - start_time)
            start_time = end_time
            print(f"FPS: {fps:.1f}")
        
        # Afficher les FPS sur l'image
        cv2.putText(frame, f"FPS: {fps_counter % 30 + 1}", (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Afficher la frame
        cv2.imshow('Reconnaissance de Gestes', frame)
        
        # Quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Application fermée")

if __name__ == "__main__":
    main()