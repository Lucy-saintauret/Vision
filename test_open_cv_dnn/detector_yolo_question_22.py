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
args = parser.parse_args()

# -------------------------
# CHARGEMENT DU MODELE YOLO
# -------------------------
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")

# Charger les noms de classes COCO
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Filtre des classes
selected_classes = [cls for cls in classes if cls in args.classes] if args.classes else classes

# -------------------------
# PARAMETRES DETECTION
# -------------------------
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

    # Suppression des doublons
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    objects = []
    for i in indices:
        i = i[0] if isinstance(i, (list, np.ndarray)) else i
        obj = {
            "class": classes[class_ids[i]],
            "confidence": float(confidences[i]),
            "bbox": boxes[i]
        }
        objects.append(obj)

        # Dessiner sur l'image
        x, y, w, h = boxes[i]
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{classes[class_ids[i]]} {confidences[i]:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame, objects

# -------------------------
# TRAITEMENT IMAGE OU WEBCAM
# -------------------------
data_output = {"source": None, "objects": []}

if args.image:
    image = cv2.imread(args.image)
    output, objects = detect_objects(image)
    data_output["source"] = os.path.basename(args.image)
    data_output["objects"] = objects

    # Afficher l'image avec détections
    cv2.imshow("Object Detection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif args.webcam:
    data_output["source"] = "webcam"
    class_counts = {}  # dictionnaire pour compter le nombre d'objets par classe
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output, objects = detect_objects(frame)

        # Mettre à jour le nombre maximal observé pour chaque classe
        current_counts = {}
        for obj in objects:
            cls = obj["class"]
            current_counts[cls] = current_counts.get(cls, 0) + 1

        for cls, count in current_counts.items():
            if cls not in class_counts or count > class_counts[cls]:
                class_counts[cls] = count

        cv2.imshow("Object Detection Webcam", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Créer la liste finale des objets détectés avec leur nombre
    data_output["objects"] = [{"class": cls, "count": count} for cls, count in class_counts.items()]

else:
    print("Veuillez spécifier une image (--image chemin) ou activer la webcam (--webcam)")
    exit()

# -------------------------
# ECRITURE DU JSON
# -------------------------
output_file = getattr(args, "output", "output.json")
with open(output_file, "w") as f:
    json.dump(data_output, f, indent=4)

print(f"Résultat JSON enregistré dans {output_file}")
