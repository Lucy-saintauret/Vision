import argparse
import cv2
import numpy as np

# Classes (Remplacez-les par celles que vous voulez utiliser)
CLASSES = [
    'banane', 'pomme', 'raisin', 'bouteille', 'broccoli', 'orange'
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

    # Preprocess the image and prepare blob for model (using fixed 640x640 size)
    blob = cv2.dnn.blobFromImage(original_image, scalefactor=1/255, size=(640, 640), swapRB=True)
    model.setInput(blob)

    # Perform inference
    outputs = model.forward()

    # Reshape output
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    # Calculate scaling factors to map coordinates back to original image
    scale_x = width / 640.0
    scale_y = height / 640.0

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

    # Draw bounding boxes on the image with proper scaling
    for i in range(len(boxes)):
        box = boxes[i]
        # Scale coordinates back to original image size
        x = round(box[0] * scale_x)
        y = round(box[1] * scale_y)
        w = round(box[2] * scale_x)
        h = round(box[3] * scale_y)
        
        draw_bounding_box(original_image, class_ids[i], scores[i],
                          x, y, x + w, y + h)

    # Resize image for display if it's too large
    display_image = original_image.copy()
    max_display_size = 800  # Taille maximale pour l'affichage
    
    if max(height, width) > max_display_size:
        if width > height:
            new_width = max_display_size
            new_height = int(height * (max_display_size / width))
        else:
            new_height = max_display_size
            new_width = int(width * (max_display_size / height))
        
        display_image = cv2.resize(display_image, (new_width, new_height))
    
    # Display the image with bounding boxes
    cv2.imshow('image', display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov5nu.onnx', help='Input your ONNX model.')
    parser.add_argument('--img', default='eagle.jpg', help='Path to input image.')
    args = parser.parse_args()
    main(args.model, args.img)
