import argparse
import cv2
import numpy as np

# Classes (Remplacez-les par celles que vous voulez utiliser)
CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    Draws bounding boxes on the input image based on the provided arguments.
    """
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main(onnx_model, input_image):
    """
    Main function to load ONNX model, perform inference, draw bounding boxes, and display the output image.
    """
    # Load the ONNX model
    model = cv2.dnn.readNetFromONNX(onnx_model)

    # Read the input image
    original_image = cv2.imread(input_image)
    height, width = original_image.shape[:2]

    # Prepare a square image for inference
    length = max(height, width)
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image

    # Preprocess the image and prepare blob for model
    blob = cv2.dnn.blobFromImage(image, scalefactor=1/255, size=(640, 640), swapRB=True)
    model.setInput(blob)

    # Perform inference
    outputs = model.forward()

    # Reshape output
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    # Iterate through output to collect bounding boxes, confidence scores, and class IDs
    for i in range(rows):
        class_scores = outputs[0][i][4:]
        max_score = max(class_scores)
        max_class_id = np.argmax(class_scores)
        
        if max_score >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]), 
                outputs[0][i][1] - (0.5 * outputs[0][i][3]), 
                outputs[0][i][2], outputs[0][i][3]
            ]
            boxes.append(box)
            scores.append(max_score)
            class_ids.append(max_class_id)

    # Draw bounding boxes on the image
    for i in range(len(boxes)):
        box = boxes[i]
        draw_bounding_box(original_image, class_ids[i], scores[i],
                          round(box[0]), round(box[1]),
                          round(box[0] + box[2]), round(box[1] + box[3]))

    # Display the image with bounding boxes
    cv2.imshow('image', original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov5nu.onnx', help='Input your ONNX model.')
    parser.add_argument('--img', default='eagle.jpg', help='Path to input image.')
    args = parser.parse_args()
    main(args.model, args.img)
