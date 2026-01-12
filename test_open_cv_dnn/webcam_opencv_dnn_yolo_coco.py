#!/usr/bin/env python3

"""webcam_opencv_dnn_yolo_coco.py: test of opencv yolo with a webcam"""

__author__      = "Fabrice Jumel"
__license__ = "CC0"
__version__ = "0.1"

import cv2
import numpy as np

# Load YOLOv3 model with OpenCV
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Initialize webcam capture
cap = cv2.VideoCapture(0)  # 0 corresponds to the default webcam (change as needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600);
# cap.set(cv2.CAP_PROP_FPS,120)
# example resolution logi hd1080 : 2304x1536 2304x1296 1920x1080 1600x896 1280x720 960x720 1024x576 800x600 864x480 800x448 640x480 640x360 432x240 352x288 320x240 320x180 176x144 160x120 160x90
# example resolution logi 720 : 1280x960 1280x720 1184x656 960x720 1024x576 800x600 864x480 800x448 752x416 640x480 640x360 432x240 352x288 320x240 320x176 176x144 160x120 160x90
while True:
    ret, frame = cap.read()  # Read a frame from the webcam

    if not ret:
        break

    # Get the height and width of the frame
    height, width = frame.shape[:2]

    # Create a blob from the frame (preprocess)
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Set the input to the neural network
    net.setInput(blob)

    # Get the output layer names
    layer_names = net.getUnconnectedOutLayersNames()

    # Run forward pass through the network
    outs = net.forward(layer_names)

    # Initialize lists to store detected objects' class IDs, confidences, and bounding boxes
    class_ids = []
    confidences = []
    boxes = []

    # Minimum confidence threshold for object detection
    conf_threshold = 0.5

    # Iterate through each output layer
    for out in outs:
        # Iterate through each detection in the output
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                # Scale the bounding box coordinates to the original frame size
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate the top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Non-maximum suppression to remove redundant boxes
    nms_threshold = 0.4
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Draw bounding boxes and labels on the frame
    for i in indices:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0)  # Green color for the bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame with detected objects
    cv2.imshow("Object Detection", frame)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

