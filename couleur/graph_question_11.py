import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Fonction pour mesurer le diamètre en pixels ---
def measure_diameter(image_path, lower_hsv, upper_hsv):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image introuvable: {image_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    return 2 * radius # diamètre en pixels


# --- Calibration avec modèle inverse ---
def calibrate_inverse(image_paths, known_distances_cm, lower_hsv, upper_hsv):
    diameters = []

    for path in image_paths:
        d = measure_diameter(path, lower_hsv, upper_hsv)
        if d:
            diameters.append(d)
            print(f"{path}: {d:.1f}px")
        else:
            diameters.append(0)
            print(f"{path}: pas de détection")

    diameters = np.array(diameters)
    distances = np.array(known_distances_cm)

    # Ajustement du modèle distance = alpha/pixels + beta
    X = np.vstack([1/diameters, np.ones_like(diameters)]).T
    coeffs, _, _, _ = np.linalg.lstsq(X, distances, rcond=None)
    alpha, beta = coeffs

    print("\nCalibration : Distance ≈ alpha/pixels + beta")
    print(f"alpha = {alpha:.4f}, beta = {beta:.4f}")

    # Tracé
    plt.scatter(diameters, distances, label="Mesures")
    x_fit = np.linspace(min(diameters), max(diameters), 100)
    y_fit = alpha/x_fit + beta
    plt.plot(x_fit, y_fit, 'r-', label="Ajustement inverse")
    plt.xlabel("Diamètre apparent (pixels)")
    plt.ylabel("Distance (cm)")
    plt.title("Calibration distance vs pixels (modèle inverse)")
    plt.legend()
    plt.grid()
    plt.show()

    return alpha, beta


# --- Estimation ---
def estimate_distance_inverse(test_image_path, alpha, beta, lower_hsv, upper_hsv):
    d_px = measure_diameter(test_image_path, lower_hsv, upper_hsv)
    if d_px:
        estimated_dist = alpha/d_px + beta
        print(f"\nImage {test_image_path}: {d_px:.1f}px -> {estimated_dist:.1f} cm")
        return estimated_dist
    else:
        print(f"\nImage {test_image_path}: pas de balle détectée")
        return None


# --- MAIN ---
if __name__ == "__main__":
 # ⚠️ Adapter selon ton vert exact
 lower_green = np.array([40, 50, 50])
 upper_green = np.array([80, 255, 255])

 folder = "Photos"

 calibration_images = [
 os.path.join(folder, "photo13.jpg"),
 os.path.join(folder, "photo20.jpg"),
 os.path.join(folder, "photo37.jpg"),
 os.path.join(folder, "photo70.jpg"),
 os.path.join(folder, "photo85.jpg"),
 ]
 known_distances = [13, 20, 37, 70, 85]

 # Calibration inverse
 alpha, beta = calibrate_inverse(calibration_images, known_distances, lower_green, upper_green)

 # Test
 test_image = os.path.join(folder, "photo100.jpg")
 estimate_distance_inverse(test_image, alpha, beta, lower_green, upper_green)