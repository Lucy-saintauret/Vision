# Introduction - Vision par Ordinateur pour Robots

**Auteure** : Lucy SAINT-AURET  
**Date** : 03/10/2025  
**Professeur encadrant** : Fabrice JUMEL  
**École** : CPE Lyon  
**Module** : Vision  
**Outils / Langage** : Python 3.11.9, OpenCV 4.9, MediaPipe



## Sommaire 

- [Introduction - Vision par Ordinateur pour Robots](#introduction---vision-par-ordinateur-pour-robots)
  - [Sommaire](#sommaire)
  - [Introduction](#introduction)
  - [Tableau des Solutions Existantes](#tableau-des-solutions-existantes)
  - [Solutions Développées et Testées](#solutions-développées-et-testées)
    - [1. Reconnaissance de Gestes avec MediaPipe (Solution approfondie)](#1-reconnaissance-de-gestes-avec-mediapipe-solution-approfondie)
    - [2. Détection de Poses Corporelles avec MediaPipe](#2-détection-de-poses-corporelles-avec-mediapipe)
    - [3. Détection d'Émotions Faciales avec Calibration Adaptative](#3-détection-démotions-faciales-avec-calibration-adaptative)
  - [Installation des Dépendances](#installation-des-dépendances)
  - [Utilisation](#utilisation)
  - [Conclusion](#conclusion)


## Introduction

La vision par ordinateur pour un robot est cruciale pour permettre à la machine d'interagir intelligemment avec son environnement. Les besoins en vision varient selon les types de robots (robots de service, robots industriels, drones, véhicules autonomes,  etc.) et leurs applications spécifiques (navigation, manipulation d'objets, surveillance, etc.).

## Tableau des Solutions Existantes

| **Besoin** | **Solution/Librairie** | **Taille** | **Dépendances** | **Format Entrée** | **Format Sortie** | **Temps Traitement** | **Lien** |
|------------|------------------------|------------|-----------------|-------------------|-------------------|---------------------|----------|
| **2. Suivi d'objets** | OpenCV Tracker (CSRT, KCF) | ~20MB | OpenCV, NumPy | Image/Vidéo (BGR) | Bounding box (x,y,w,h) | 5-20ms/frame | [OpenCV Tracking](https://docs.opencv.org/4.x/d9/df8/group__tracking.html) |
| **2. Suivi d'objets** | DeepSORT | ~50MB | TensorFlow, OpenCV | Vidéo + détections | ID + trajectoires | 30-50ms/frame | [DeepSORT](https://github.com/nwojke/deep_sort) |
| **5. Navigation visuelle** | ORB-SLAM3 | ~100MB | OpenCV, Eigen3, Pangolin | Caméra RGB/RGBD | Pose + carte 3D | 20-30ms/frame | [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) |
| **5. Navigation visuelle** | OpenCV FAST/ORB | ~15MB | OpenCV | Image BGR | Points clés + descripteurs | 5-10ms/frame | [Feature Detection](https://docs.opencv.org/4.x/df/d0c/tutorial_py_fast.html) |
| **7. Reconnaissance gestes** | MediaPipe Hands | ~30MB | MediaPipe, OpenCV | Image/Vidéo RGB | 21 points 3D main | 15-25ms/frame | [MediaPipe Hands](https://mediapipe.dev/solutions/hands) |
| **7. Reconnaissance gestes** | OpenPose | ~200MB | OpenPose, CUDA (opt) | Image RGB | Points squelette | 50-200ms/frame | [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) |
| **8. OCR/Texte** | Tesseract + OpenCV | ~50MB | Tesseract, OpenCV | Image | Texte + coordonnées | 100-500ms/image | [Tesseract](https://github.com/tesseract-ocr/tesseract) |
| **8. OCR/Texte** | EasyOCR | ~100MB | PyTorch, PIL | Image | Texte + bbox + confiance | 200-800ms/image | [EasyOCR](https://github.com/JaidedAI/EasyOCR) |
| **9. Reconnaissance scènes** | TensorFlow MobileNet | ~50MB | TensorFlow | Image 224x224 | Probabilités classes | 20-50ms/image | [MobileNet](https://www.tensorflow.org/lite/models/image_classification/overview) |
| **9. Reconnaissance scènes** | CLIP (OpenAI) | ~500MB | PyTorch, transformers | Image + texte | Similarité sémantique | 100-300ms/image | [CLIP](https://github.com/openai/CLIP) |
| **11. Reconnaissance visages** | Face Recognition | ~20MB | dlib, face_recognition | Image RGB | Encodages + noms | 50-150ms/visage | [Face Recognition](https://github.com/ageitgey/face_recognition) |
| **11. Reconnaissance visages** | InsightFace | ~100MB | MXNet/ONNX | Image | Embeddings + attributs | 10-30ms/visage | [InsightFace](https://github.com/deepinsight/insightface) |
| **13. Interaction H-M** | MediaPipe Face Mesh | ~25MB | MediaPipe | Image/Vidéo | 468 points visage | 10-20ms/frame | [Face Mesh](https://mediapipe.dev/solutions/face_mesh) |
| **13. Interaction H-M** | EmotiW/FER | ~80MB | TensorFlow/PyTorch | Image visage | Émotions + scores | 30-80ms/image | [FER](https://github.com/justinshenk/fer) |
| **14. Détection anomalies** | PyOD | ~10MB | scikit-learn, NumPy | Données tabulaires | Scores anomalie | Variable | [PyOD](https://github.com/yzhao062/pyod) |
| **14. Estimation profondeur** | MiDaS | ~100MB | PyTorch, OpenCV | Image RGB | Carte profondeur | 100-300ms/image | [MiDaS](https://github.com/isl-org/MiDaS) |

## Solutions Développées et Testées

### 1. Reconnaissance de Gestes avec MediaPipe (Solution approfondie)

**Description :** Utilisation de MediaPipe pour détecter et reconnaître des gestes de la main en temps réel via webcam.

**Avantages :**
- Détection précise de 21 points de la main
- Temps de traitement rapide (15-25ms/frame)
- Fonctionne avec une simple webcam
- Robuste aux différentes conditions d'éclairage

**Limites :**
- Limité aux gestes de la main
- Peut avoir des difficultés avec des mains très éloignées
- Sensible aux occlusions partielles

**Implémentation :** [gesture_recognition.py](gesture_recognition.py)

**Explication :**

Le script ouvre la webcam, convertit chaque image en RGB et passe MediaPipe Hands pour obtenir 21 repères par main. Les coordonnées sont projetées en pixels et quelques règles géométriques comparent l’orientation et l’ouverture des doigts afin de reconnaître des gestes courants (main ouverte, poing, pouce levé, OK). Un léger lissage temporel stabilise l’étiquette affichée et les landmarks sont dessinés sur l’image.


**Vidéos exemple :** 

![gesture detection](Images/gesture_detection.gif)

### 2. Détection de Poses Corporelles avec MediaPipe

**Description :** Détection en temps réel des poses corporelles humaines pour l'interaction homme-robot.

**Avantages :**
- Détection de 33 points du corps
- Estimation 3D des poses
- Compatible webcam standard
- Bibliothèque légère et optimisée

**Limites :**
- Limité à une personne par frame
- Précision réduite pour les poses complexes
- Sensible à l'occultation

**Implémentation :** [pose_detection.py](pose_detection.py)

**Explication :**

Le script utilise MediaPipe Pose pour estimer environ 33 repères corporels à chaque frame. À partir des articulations clés (épaules, coudes, poignets, hanches), il calcule des angles ou des distances simples pour caractériser la posture ou le mouvement, puis superpose le squelette à l’image. Une résolution modérée (par ex. 640×480) et un lissage léger permettent de conserver un bon framerate tout en limitant les fluctuations.


**Vidéos exemple :**

![pose detection](Images/pose_detection.gif)

### 3. Détection d'Émotions Faciales avec Calibration Adaptative

**Description :** Système avancé de détection et reconnaissance d'émotions en temps réel avec calibration personnalisée pour chaque utilisateur.

**Avantages :**
- Utilise MediaPipe pour une détection précise des landmarks faciaux (468 points)
- Système de calibration automatique pour s'adapter au visage de l'utilisateur
- Détection de 7 émotions : Neutre, Heureux, Surpris, Triste, En colère, Dégoût, Peur
- Anti-oscillation intelligent pour éviter les changements erratiques
- Mode debug intégré pour analyser les paramètres en temps réel
- Traitement rapide (10-20ms/frame) avec lissage temporel

**Limites :**
- Nécessite une calibration initiale pour une précision optimale
- Performance dépendante de la qualité de la webcam
- Sensible aux conditions d'éclairage extrêmes
- Optimisé pour un visage par frame

**Innovations techniques :**
- **Calibration adaptative** : Enregistre les caractéristiques neutres de l'utilisateur
- **Seuils dynamiques** : Ajustement automatique selon la baseline personnelle
- **Détection multi-critères** : Combine ouverture de bouche, courbure et position des coins
- **Système anti-oscillation** : Historique des émotions pour stabiliser la détection

**Explication technique :**

Le système utilise MediaPipe Face Mesh pour extraire 468 landmarks faciaux avec une précision sub-pixel. Après calibration de l'expression neutre de l'utilisateur (1 seconde d'échantillonnage), il calcule des ratios adaptatifs :
- **MAR** (Mouth Aspect Ratio) : rapport hauteur/largeur de la bouche
- **EAR** (Eye Aspect Ratio) : degré d'ouverture des yeux  
- **Courbure buccale** : analyse des coins et du centre de la bouche
- **Position des sourcils** : distance relative aux yeux

Un algorithme de détection multi-critères nécessite la validation de plusieurs paramètres simultanés pour éviter les faux positifs. Le système anti-oscillation analyse l'historique des 3 dernières détections et favorise la stabilité lors de transitions ambiguës entre émotions proches.pour un robot est cruciale pour permettre à la machine d'interagir intelligemment avec son environnement. Les besoins en vision varient selon les types de robots (robots de service, robots industriels, drones, véhicules autonomes, etc.) et leurs applications spécifiques (navigation, manipulation d'objets, surveillance, etc.).

**Implémentation :** [emotion_detection.py](emotion_detection.py)

**Vidéo exemple :**

![emotion detection](Images/emotion_detection.gif)

Donc je commence d'abord par calibrer au neutre car quand je lance le programme il dit que je suis heureuse, et ensuite je teste plusieurs émotions.

## Installation des Dépendances

```bash
# Pour les 3 solutions développées
pip install opencv-python mediapipe numpy matplotlib

# Installation simplifiée
pip install -r requirements.txt
```

## Utilisation

```bash
# Reconnaissance de gestes
python gesture_recognition.py

# Détection de poses
python pose_detection.py

# Détection d'émotions
python emotion_detection.py
```

## Conclusion

Ces solutions offrent un bon panel des capacités de vision par ordinateur pour la robotique, avec des approches complémentaires :
- **Interaction naturelle** : reconnaissance de gestes et poses
- **Analyse émotionnelle** : détection d'émotions pour l'interaction sociale
- **Temps réel** : toutes les solutions sont optimisées pour la webcam

Chaque solution peut être étendue et adaptée selon les besoins spécifiques du robot et de son environnement d'application.
