#!/usr/bin/env python3
"""
Test de validation pour toutes les solutions de vision
====================================================

Ce script teste rapidement que toutes les solutions peuvent être importées
et initialisées correctement.

Utilisation : python test_solutions.py
"""

def test_gesture_recognition():
    """Test de la reconnaissance de gestes"""
    print("🤚 Test de la reconnaissance de gestes...")
    try:
        import gesture_recognition
        recognizer = gesture_recognition.GestureRecognizer()
        print("✓ Reconnaissance de gestes : OK")
        return True
    except Exception as e:
        print(f"✗ Reconnaissance de gestes : ERREUR - {e}")
        return False

def test_pose_detection():
    """Test de la détection de poses"""
    print("🧍 Test de la détection de poses...")
    try:
        import pose_detection
        detector = pose_detection.PoseDetector()
        print("✓ Détection de poses : OK")
        return True
    except Exception as e:
        print(f"✗ Détection de poses : ERREUR - {e}")
        return False

def test_emotion_detection():
    """Test de la détection d'émotions"""
    print("😊 Test de la détection d'émotions...")
    try:
        import emotion_detection
        detector = emotion_detection.EmotionDetector()
        print("✓ Détection d'émotions : OK")
        return True
    except Exception as e:
        print(f"✗ Détection d'émotions : ERREUR - {e}")
        return False

def test_dependencies():
    """Test des dépendances critiques"""
    print("📦 Test des dépendances...")
    
    dependencies = [
        ("cv2", "OpenCV"),
        ("mediapipe", "MediaPipe"),
        ("numpy", "NumPy")
    ]
    
    success_count = 0
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✓ {name} : OK")
            success_count += 1
        except ImportError:
            print(f"✗ {name} : MANQUANT")
    
    return success_count == len(dependencies)

def main():
    print("=== Test de Validation des Solutions de Vision ===")
    print()
    
    # Test des dépendances
    deps_ok = test_dependencies()
    print()
    
    if not deps_ok:
        print("⚠ Certaines dépendances sont manquantes.")
        print("Lancez : pip install -r requirements.txt")
        return
    
    # Test des solutions
    tests = [
        test_gesture_recognition,
        test_pose_detection,
        test_emotion_detection
    ]
    
    success_count = 0
    for test_func in tests:
        if test_func():
            success_count += 1
        print()
    
    # Résumé
    print("=== Résumé ===")
    print(f"Solutions fonctionnelles : {success_count}/{len(tests)}")
    
    if success_count == len(tests):
        print("🎉 Toutes les solutions sont prêtes !")
        print("Lancez : python main_demo.py")
    else:
        print("⚠ Certaines solutions ont des problèmes.")
        print("Vérifiez les erreurs ci-dessus.")

if __name__ == "__main__":
    main()