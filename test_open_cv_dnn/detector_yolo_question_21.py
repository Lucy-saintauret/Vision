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
parser.add_argument("--classes", nargs="+", default=[], help="Liste des classes à détecter (ex: person car dog)")
args = parser.parse_args()

# -------------------------
# CHARGEMENT DU MODELE YOLO
# -------------------------

# Load the YOLOv3 model with OpenCV
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")

# Load the COCO class labels
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")
    
# Si l'utilisateur a fourni une liste de classes, on ne garde que celles-ci
if args.classes:
    selected_classes = [cls for cls in classes if cls in args.classes]
else:
    selected_classes = classes  # Sinon, toutes les classes sont considérées


# Initialize lists to store detected objects' class IDs, confidences, and bounding boxes
class_ids = []
confidences = []
boxes = []

# Minimum confidence threshold for object detection
conf_threshold = 0.2
# Non-maximum suppression to remove redundant boxes
nms_threshold = 0.4

# -------------------------
# FONCTION DETECTION
# -------------------------
def detect_objects(frame):
    height, width = frame.shape[:2]

    # Prétraitement de l'image
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

            # Vérifier que la détection est dans la liste des classes sélectionnées
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

    # Dessiner les boîtes
    for i in indices:
        i = i[0] if isinstance(i, (list, np.ndarray)) else i
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

# Non-maximum suppression to remove redundant boxes
nms_threshold = 0.4
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# -------------------------
# TRAITEMENT IMAGE OU WEBCAM
# -------------------------
if args.image:
    image = cv2.imread(args.image)
    output = detect_objects(image)
    cv2.imshow("Object Detection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
elif args.webcam:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = detect_objects(frame)
        cv2.imshow("Object Detection Webcam", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
else:
    print("Veuillez spécifier une image (--image chemin) ou activer la webcam (--webcam)")


