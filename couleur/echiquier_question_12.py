import cv2
import numpy as np
import glob
import os

# Dimensions du damier (nb coins intérieurs)
CHECKERBOARD = (8, 6)

# Préparation des points 3D du damier (Z=0)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = []  # points 3D
imgpoints = []  # points 2D

# Création du dossier pour sauvegarder les images avec coins détectés
output_folder = "output_corners"
os.makedirs(output_folder, exist_ok=True)

images = glob.glob("Photos/Echiquier/*.jpg")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Dessine les coins détectés
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)

        # Sauvegarde l'image annotée
        basename = os.path.basename(fname)
        save_path = os.path.join(output_folder, f"corners_{basename}")
        cv2.imwrite(save_path, img)
        print(f"[OK] Coins détectés sur {basename}, sauvegardé dans {save_path}")
    else:
        print(f"[X] Échec détection coins sur {fname}")

# Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("\nMatrice intrinsèque :\n", mtx)
print("Coefficients de distorsion :\n", dist)


# Exemple : estimer distance d'une balle avec affichage
def estimate_distance_ball(image_path, mtx, dist, real_diameter_cm):
    img = cv2.imread(image_path)
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)

    hsv = cv2.cvtColor(undistorted, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))  # vert
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        d_px = 2 * radius
        f = (mtx[0, 0] + mtx[1, 1]) / 2  # focale moyenne en pixels
        distance = (f * real_diameter_cm) / d_px

        center = (int(x), int(y))
        cv2.circle(undistorted, center, int(radius), (0, 255, 0), 2)  # cercle autour de la balle
        cv2.putText(
            undistorted,
            f"{distance:.1f} cm",
            (center[0] - 40, center[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        print(f"Diamètre: {d_px:.1f}px -> Distance estimée: {distance:.1f} cm")

    cv2.imshow("Estimation Distance", undistorted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --- TEST ---
if __name__ == "__main__":
    test_image = "Photos/photo20.jpg"  # à modifier selon ton image
    estimate_distance_ball(test_image, mtx, dist, real_diameter_cm=4.0)
