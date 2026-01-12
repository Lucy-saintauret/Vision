"""
Programme de détection d'émotions en temps réel à partir de la webcam
Utilise MediaPipe pour la détection de visage et analyse des landmarks faciaux
pour détecter les émotions de base.
"""

import cv2
import mediapipe as mp
import numpy as np
import math

class EmotionDetector:
    def __init__(self, debug_mode=False):
        # Initialisation de MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.debug_mode = debug_mode
        
        # Variables pour la calibration et le lissage
        self.calibration_mode = False
        self.calibration_samples = []
        self.neutral_baseline = {
            'mar': 0.02,
            'ear': 0.2,
            'mouth_curve': 0,
            'eyebrow_distance': 18
        }
        
        # Variables pour le lissage des émotions
        self.emotion_history = []
        self.emotion_smoothing = 3  # Nombre d'échantillons pour le lissage
        
        # Configuration du détecteur de visage
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Points de référence pour les émotions
        # Indices des landmarks MediaPipe pour différentes parties du visage
        self.LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.MOUTH = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
        self.EYEBROWS_LEFT = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
        self.EYEBROWS_RIGHT = [296, 334, 293, 300, 276, 283, 282, 295, 285, 336]
        
    def calculate_distance(self, p1, p2):
        """Calcule la distance euclidienne entre deux points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def calculate_eye_aspect_ratio(self, eye_landmarks):
        """Calcule le ratio d'aspect de l'œil pour détecter si l'œil est fermé"""
        if len(eye_landmarks) < 6:
            return 0
        
        # Distance verticale entre les points haut/bas de l'œil
        vertical_1 = self.calculate_distance(eye_landmarks[1], eye_landmarks[5])
        vertical_2 = self.calculate_distance(eye_landmarks[2], eye_landmarks[4])
        
        # Distance horizontale de l'œil
        horizontal = self.calculate_distance(eye_landmarks[0], eye_landmarks[3])
        
        if horizontal == 0:
            return 0
        
        # Calcul du ratio d'aspect
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def calculate_mouth_aspect_ratio(self, mouth_landmarks):
        """Calcule le ratio d'aspect de la bouche"""
        if len(mouth_landmarks) < 6:
            return 0
        
        # Distance verticale de la bouche
        vertical = self.calculate_distance(mouth_landmarks[1], mouth_landmarks[5])
        
        # Distance horizontale de la bouche
        horizontal = self.calculate_distance(mouth_landmarks[0], mouth_landmarks[3])
        
        if horizontal == 0:
            return 0
        
        mar = vertical / horizontal
        return mar
    
    def detect_emotion(self, landmarks, image_shape):
        """Détecte l'émotion basée sur les landmarks faciaux"""
        h, w = image_shape[:2]
        
        # Conversion des landmarks normalisés en coordonnées pixel
        face_points = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            face_points.append([x, y])
        
        # Extraction des points clés
        left_eye_points = [face_points[i] for i in self.LEFT_EYE[:6]]
        right_eye_points = [face_points[i] for i in self.RIGHT_EYE[:6]]
        mouth_points = [face_points[i] for i in self.MOUTH[:6]]
        
        # Calcul des ratios
        left_ear = self.calculate_eye_aspect_ratio(left_eye_points)
        right_ear = self.calculate_eye_aspect_ratio(right_eye_points)
        avg_ear = (left_ear + right_ear) / 2.0
        
        mar = self.calculate_mouth_aspect_ratio(mouth_points)
        
        # Points pour analyser la courbure de la bouche et les coins
        mouth_left = face_points[61]
        mouth_right = face_points[291]
        mouth_top = face_points[13]
        mouth_bottom = face_points[14]
        mouth_center = face_points[17]
        
        # Calcul de la courbure de la bouche (sourire vs froncement)
        mouth_curve = (mouth_left[1] + mouth_right[1]) / 2 - mouth_top[1]
        mouth_width = self.calculate_distance(mouth_left, mouth_right)
        
        # Calcul additionnel pour les coins de la bouche (sourire)
        left_corner_lift = mouth_center[1] - mouth_left[1]   # Coin gauche relevé
        right_corner_lift = mouth_center[1] - mouth_right[1] # Coin droit relevé
        corner_lift_avg = (left_corner_lift + right_corner_lift) / 2
        
        # Points des sourcils pour détecter la surprise ou la colère
        left_eyebrow = [face_points[i] for i in self.EYEBROWS_LEFT[:3]]
        right_eyebrow = [face_points[i] for i in self.EYEBROWS_RIGHT[:3]]
        
        # Hauteur moyenne des sourcils
        eyebrow_height = (sum([p[1] for p in left_eyebrow + right_eyebrow]) / 6)
        eye_height = (sum([face_points[i][1] for i in [33, 362]]) / 2)
        eyebrow_eye_distance = eye_height - eyebrow_height
        
        # Calibration si nécessaire
        calibration_done = self.calibrate_neutral(mar, avg_ear, mouth_curve, eyebrow_eye_distance)
        
        if self.calibration_mode:
            emotion = "Calibration en cours..."
            confidence = len(self.calibration_samples) / 30.0
            return emotion, confidence
        
        # Utiliser la baseline calibrée pour ajuster les seuils
        baseline_mar = self.neutral_baseline['mar']
        baseline_ear = self.neutral_baseline['ear']
        baseline_curve = self.neutral_baseline['mouth_curve']
        baseline_eyebrow = self.neutral_baseline['eyebrow_distance']
        
        # Seuils adaptatifs basés sur la calibration (un tout petit peu plus strict)
        SMILE_THRESHOLD = baseline_mar + 0.016        # Juste un tout petit peu plus strict
        MOUTH_CURVE_SMILE = baseline_curve - 7        # Un peu plus prononcé nécessaire
        SURPRISE_EAR_THRESHOLD = baseline_ear + 0.15
        SURPRISE_EYEBROW_THRESHOLD = baseline_eyebrow + 12
        ANGER_EYEBROW_THRESHOLD = baseline_eyebrow - 8
        
        # Détection du sourire (un tout petit peu plus stricte)
        smile_mar_condition = mar > SMILE_THRESHOLD
        smile_curve_condition = mouth_curve < MOUTH_CURVE_SMILE
        smile_width_check = mouth_width > (baseline_ear * 110)  
        smile_corners_condition = corner_lift_avg > 4  # Un petit peu plus élevé
        
        # Conditions équilibrées : au moins 2 critères principaux
        smile_criteria_count = sum([
            smile_mar_condition,
            smile_curve_condition, 
            smile_corners_condition,
            mar > (baseline_mar + 0.010)  # Seuil minimal un peu plus élevé
        ])
        
        # 2 critères avec au moins l'ouverture de bouche OU les coins relevés
        if smile_criteria_count >= 2 and (smile_mar_condition or smile_corners_condition):
            emotion = "Heureux"
            smile_intensity = max((mar - baseline_mar) * 28, 
                                (baseline_curve - mouth_curve) * 0.07,
                                corner_lift_avg * 0.035)
            confidence = min(0.95, 0.78 + smile_intensity)
        
        # Détection de la surprise (basée sur la calibration)
        elif avg_ear > SURPRISE_EAR_THRESHOLD and eyebrow_eye_distance > SURPRISE_EYEBROW_THRESHOLD and mar > (baseline_mar + 0.005):
            emotion = "Surpris"
            confidence = min(0.9, 0.6 + (avg_ear - SURPRISE_EAR_THRESHOLD) * 4)
        
        # Détection de la colère (sourcils très froncés, bouche fermée)
        elif eyebrow_eye_distance < ANGER_EYEBROW_THRESHOLD and mar < (baseline_mar - 0.005):
            emotion = "En colère"
            confidence = min(0.85, 0.6 + (ANGER_EYEBROW_THRESHOLD - eyebrow_eye_distance) / 10)
        
        # Détection de la tristesse
        elif mar < (baseline_mar - 0.008) and mouth_curve > (baseline_curve + 5):
            emotion = "Triste"
            confidence = min(0.8, 0.6 + (baseline_mar - mar) * 40)
        
        # Détection du dégoût
        elif mar > (baseline_mar + 0.01) and mouth_curve > (baseline_curve + 3) and avg_ear < (baseline_ear - 0.02):
            emotion = "Dégoût"
            confidence = min(0.75, 0.6 + (mar - baseline_mar) * 15)
        
        # Détection de la peur (yeux très écarquillés, bouche légèrement ouverte)
        elif avg_ear > (baseline_ear + 0.12) and mar > (baseline_mar + 0.008) and mar < (baseline_mar + 0.025) and eyebrow_eye_distance > (baseline_eyebrow + 8):
            emotion = "Peur"
            confidence = min(0.8, 0.6 + (avg_ear - baseline_ear) * 3)
        
        # Amélioration de la détection neutre (zones plus stables)
        else:
            # Tolérance réduite pour une zone neutre plus stable
            tolerance_mar = 0.008        # Réduit de 0.01 à 0.008
            tolerance_ear = 0.06         # Réduit de 0.08 à 0.06
            tolerance_curve = 3          # Réduit de 4 à 3
            tolerance_eyebrow = 6        # Réduit de 8 à 6
            
            is_neutral_mouth = (baseline_mar - tolerance_mar) < mar < (baseline_mar + tolerance_mar) and \
                              (baseline_curve - tolerance_curve) < mouth_curve < (baseline_curve + tolerance_curve)
            is_neutral_eyes = (baseline_ear - tolerance_ear) < avg_ear < (baseline_ear + tolerance_ear)
            is_neutral_eyebrows = (baseline_eyebrow - tolerance_eyebrow) < eyebrow_eye_distance < (baseline_eyebrow + tolerance_eyebrow)
            is_neutral_corners = abs(corner_lift_avg) < 2  # Coins pas trop relevés
            
            if is_neutral_mouth and is_neutral_eyes and is_neutral_eyebrows and is_neutral_corners:
                emotion = "Neutre"
                confidence = 0.9
            else:
                emotion = "Neutre"
                confidence = 0.65  # Confiance réduite si pas parfaitement neutre
        
        # Mode débogage pour afficher les valeurs (avec infos sourire)
        if self.debug_mode:
            print(f"Debug - MAR: {mar:.3f} (baseline: {baseline_mar:.3f}), "
                  f"EAR: {avg_ear:.3f} (baseline: {baseline_ear:.3f})")
            print(f"      - Mouth_curve: {mouth_curve:.1f} (baseline: {baseline_curve:.1f}), "
                  f"Eyebrow_dist: {eyebrow_eye_distance:.1f} (baseline: {baseline_eyebrow:.1f})")
            print(f"      - Smile conditions: MAR>{SMILE_THRESHOLD:.3f}? {mar > SMILE_THRESHOLD}, "
                  f"Curve<{MOUTH_CURVE_SMILE:.1f}? {mouth_curve < MOUTH_CURVE_SMILE}, "
                  f"Corners: {corner_lift_avg:.1f}, Criteria: {smile_criteria_count}")
            print(f"      - Emotion: {emotion}, Confidence: {confidence:.2f}")
            print("---")
        
        # Lissage des émotions pour éviter l'oscillation (équilibré)
        self.emotion_history.append(emotion)
        if len(self.emotion_history) > self.emotion_smoothing:
            self.emotion_history.pop(0)
        
        # Système anti-oscillation équilibré
        if len(self.emotion_history) >= 3:
            recent_emotions = self.emotion_history[-3:]
            
            # Si on oscille entre neutre et heureux, favoriser neutre modérément
            if set(recent_emotions) == {"Neutre", "Heureux"}:
                if confidence < 0.87:  # Seuil intermédiaire
                    emotion = "Neutre"
                    confidence = 0.82
            
            # Si 2 des 3 dernières sont neutres, être un peu plus exigeant
            elif recent_emotions.count("Neutre") >= 2 and emotion == "Heureux":
                if confidence < 0.85:  # Seuil raisonnable
                    emotion = "Neutre"
                    confidence = 0.85
        
        return emotion, confidence
    
    def calibrate_neutral(self, mar, avg_ear, mouth_curve, eyebrow_eye_distance):
        """Calibre l'expression neutre de l'utilisateur"""
        if self.calibration_mode:
            sample = {
                'mar': mar,
                'ear': avg_ear,
                'mouth_curve': mouth_curve,
                'eyebrow_distance': eyebrow_eye_distance
            }
            self.calibration_samples.append(sample)
            
            if len(self.calibration_samples) >= 30:  # 30 échantillons (environ 1 seconde)
                # Calculer la moyenne
                self.neutral_baseline['mar'] = sum(s['mar'] for s in self.calibration_samples) / len(self.calibration_samples)
                self.neutral_baseline['ear'] = sum(s['ear'] for s in self.calibration_samples) / len(self.calibration_samples)
                self.neutral_baseline['mouth_curve'] = sum(s['mouth_curve'] for s in self.calibration_samples) / len(self.calibration_samples)
                self.neutral_baseline['eyebrow_distance'] = sum(s['eyebrow_distance'] for s in self.calibration_samples) / len(self.calibration_samples)
                
                self.calibration_mode = False
                self.calibration_samples = []
                print(f"Calibration terminée! Nouvelle baseline: {self.neutral_baseline}")
                return True
        return False
    
    def draw_emotion_info(self, image, emotion, confidence, landmarks):
        """Dessine les informations d'émotion sur l'image"""
        h, w = image.shape[:2]
        
        # Dessiner les landmarks du visage
        if landmarks:
            self.mp_drawing.draw_landmarks(
                image, 
                landmarks,
                self.mp_face_mesh.FACEMESH_CONTOURS,
                None,
                self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
        
        # Couleur basée sur l'émotion
        color_map = {
            "Heureux": (0, 255, 0),      # Vert
            "Triste": (255, 0, 0),       # Bleu
            "Surpris": (0, 255, 255),    # Jaune
            "En colère": (0, 0, 255),    # Rouge
            "Dégoût": (128, 0, 128),     # Violet
            "Peur": (255, 165, 0),       # Orange
            "Neutre": (255, 255, 255)    # Blanc
        }
        
        color = color_map.get(emotion, (255, 255, 255))
        
        # Dessiner un rectangle pour le texte
        cv2.rectangle(image, (10, 10), (400, 100), (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (400, 100), color, 2)
        
        # Afficher l'émotion et la confiance
        cv2.putText(image, f"Emotion: {emotion}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(image, f"Confiance: {confidence:.2f}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Affichage des instructions pour basculer le mode debug
        if self.debug_mode:
            cv2.putText(image, "Mode DEBUG (appuyez sur 'd' pour desactiver)", (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        else:
            cv2.putText(image, "Appuyez sur 'd' pour debug, 'c' pour calibrer", (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Affichage spécial pour la calibration
        if self.calibration_mode:
            progress = len(self.calibration_samples) / 30.0 * 100
            cv2.putText(image, f"CALIBRATION: Restez neutre! {progress:.0f}%", (20, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return image
    
    def run(self):
        """Lance la détection d'émotions en temps réel"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Erreur: Impossible d'ouvrir la webcam")
            return
        
        print("Détection d'émotions démarrée. Appuyez sur 'q' pour quitter.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erreur: Impossible de lire l'image de la webcam")
                break
            
            # Flip horizontal pour effet miroir
            frame = cv2.flip(frame, 1)
            
            # Conversion en RGB pour MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Détection des landmarks faciaux
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Détection de l'émotion
                    emotion, confidence = self.detect_emotion(face_landmarks, frame.shape)
                    
                    # Dessiner les informations sur l'image
                    frame = self.draw_emotion_info(frame, emotion, confidence, face_landmarks)
            else:
                # Aucun visage détecté
                cv2.putText(frame, "Aucun visage detecte", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Affichage des instructions
            cv2.putText(frame, "q: quitter, d: debug, c: calibrer neutre", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Affichage de l'image
            cv2.imshow('Detection d\'Emotions', frame)
            
            # Gestion des touches
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                print(f"Mode debug {'activé' if self.debug_mode else 'désactivé'}")
            elif key == ord('c'):
                if not self.calibration_mode:
                    self.calibration_mode = True
                    self.calibration_samples = []
                    print("Calibration démarrée! Gardez une expression neutre pendant 1 seconde...")
        
        # Nettoyage
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Fonction principale"""
    detector = EmotionDetector()
    detector.run()

if __name__ == "__main__":
    main()