import cv2
import numpy as np
import argparse
import json
import os

# -------------------------
# ARGUMENTS LIGNE DE COMMANDE
# -------------------------
parser = argparse.ArgumentParser(description="YOLOv3-Tiny Object Detection")
parser.add_argument("--image", help="Chemin de l'image à traiter")
parser.add_argument("--webcam", action="store_true", help="Activer le flux webcam")
parser.add_argument("--classes", nargs="+", default=[], help="Liste des classes à détecter (ex: person dog)")
parser.add_argument("--output", default="output.json", help="Nom du fichier JSON de sortie")
parser.add_argument("--save_crops", default="crops", help="Dossier pour sauvegarder les objets détectés")
parser.add_argument("--mosaic_size", type=int, default=200, help="Taille d'une image dans la mosaïque (px)")
args = parser.parse_args()

# -------------------------
# CREER DOSSIER POUR CROPS
# -------------------------
os.makedirs(args.save_crops, exist_ok=True)

# -------------------------
# CHARGEMENT DU MODELE YOLO
# -------------------------
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")

with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

selected_classes = [cls for cls in classes if cls in args.classes] if args.classes else classes

conf_threshold = 0.2
nms_threshold = 0.4

# -------------------------
# FONCTION DETECTION
# -------------------------
def detect_objects(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    outs = net.forward(layer_names)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold and classes[class_id] in selected_classes:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    objects = []
    for i in indices:
        i = i[0] if isinstance(i, (list, np.ndarray)) else i
        obj_class = classes[class_ids[i]]
        obj = {
            "class": obj_class,
            "confidence": float(confidences[i]),
            "bbox": boxes[i]
        }
        objects.append(obj)

        # Dessiner la boîte
        x, y, w, h = boxes[i]
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{obj_class} {confidences[i]:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Sauvegarder le crop
        crop_img = frame[y:y+h, x:x+w]
        class_folder = os.path.join(args.save_crops, obj_class)
        os.makedirs(class_folder, exist_ok=True)
        count = len(os.listdir(class_folder))
        crop_path = os.path.join(class_folder, f"{count+1}.jpg")
        cv2.imwrite(crop_path, crop_img)

    return frame, objects

# -------------------------
# FONCTION MOSAIQUE
# -------------------------
def create_mosaic(image_folder, mosaic_size=200):
    # Récupérer toutes les images de la classe
    images = [cv2.imread(os.path.join(image_folder, f)) for f in os.listdir(image_folder) if f.endswith(".jpg")]
    if not images:
        return None

    # Redimensionner toutes les images à mosaic_size
    resized = [cv2.resize(img, (mosaic_size, mosaic_size)) for img in images]

    # Calculer nombre de colonnes et lignes
    cols = int(np.ceil(np.sqrt(len(resized))))
    rows = int(np.ceil(len(resized)/cols))

    # Créer l'image finale
    mosaic = np.zeros((rows*mosaic_size, cols*mosaic_size, 3), dtype=np.uint8)

    for idx, img in enumerate(resized):
        y = (idx // cols) * mosaic_size
        x = (idx % cols) * mosaic_size
        mosaic[y:y+mosaic_size, x:x+mosaic_size] = img

    return mosaic

# -------------------------
# TRAITEMENT IMAGE OU WEBCAM
# -------------------------
data_output = {"source": None, "objects": []}
class_counts = {}

if args.image:
    image = cv2.imread(args.image)
    output, objects = detect_objects(image)
    data_output["source"] = os.path.basename(args.image)

    for obj in objects:
        cls = obj["class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    data_output["objects"] = [{"class": cls, "count": count} for cls, count in class_counts.items()]

    cv2.imshow("Object Detection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif args.webcam:
    data_output["source"] = "webcam"
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        output, objects = detect_objects(frame)
        # Mettre à jour le nombre maximal observé pour chaque classe
        for obj in objects:
            cls = obj["class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1

        cv2.imshow("Object Detection Webcam", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    data_output["objects"] = [{"class": cls, "count": count} for cls, count in class_counts.items()]

# -------------------------
# ENREGISTRER JSON
# -------------------------
with open(args.output, "w") as f:
    json.dump(data_output, f, indent=4)
print(f"Résultat JSON enregistré dans {args.output}")

# -------------------------
# CREER MOSAIQUES PAR CLASSE
# -------------------------
for cls in selected_classes:
    class_folder = os.path.join(args.save_crops, cls)
    if os.path.exists(class_folder):
        mosaic = create_mosaic(class_folder, args.mosaic_size)
        if mosaic is not None:
            mosaic_path = os.path.join(args.save_crops, f"{cls}_mosaic.jpg")
            cv2.imwrite(mosaic_path, mosaic)
            print(f"Mosaïque pour {cls} enregistrée dans {mosaic_path}")
