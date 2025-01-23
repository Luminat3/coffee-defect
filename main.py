import onnxruntime as ort
import numpy as np
import cv2

# Load the ONNX model
session = ort.InferenceSession("model/yolov8n.onnx")

# Define the input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Preprocess the input image
def preprocess(image):
    img = cv2.resize(image, (640, 640))  # Resize to model input size
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    img = np.transpose(img, (2, 0, 1))  # Change to CHW format
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Postprocess the outputs
def postprocess(outputs, image_shape, conf_threshold=0.5, iou_threshold=0.4):
    """
    Convert raw model outputs into filtered bounding boxes.

    Args:
    - outputs: Raw model outputs
    - image_shape: Original image dimensions (height, width)
    - conf_threshold: Confidence threshold
    - iou_threshold: Intersection Over Union threshold for NMS

    Returns:
    - List of bounding boxes: [x1, y1, x2, y2, confidence, class_id]
    """
    predictions = outputs[0]  # Extract the first (and only) batch
    if predictions.shape != (1, 84, 8400):
        raise ValueError(f"Unexpected output shape: {predictions.shape}")

    predictions = predictions.squeeze(0)  # Remove batch dimension: [84, 8400]

    # Extract components
    boxes = predictions[:4, :].T  # Bounding box coordinates: [8400, 4]
    object_conf = predictions[4, :]  # Object confidence scores: [8400]
    class_probs = predictions[5:, :].T  # Class probabilities: [8400, 80]
    class_ids = np.argmax(class_probs, axis=1)  # Class IDs: [8400]
    class_conf = np.max(class_probs, axis=1)  # Class confidence scores: [8400]

    # Combine object confidence with class confidence
    scores = object_conf * class_conf

    # Filter by confidence threshold
    mask = scores > conf_threshold
    if mask.sum() == 0:
        return []  # No valid detections

    boxes = boxes[mask]  # Filter boxes
    scores = scores[mask]  # Filter scores
    class_ids = class_ids[mask]  # Filter class IDs

    # Rescale boxes to the original image size
    h, w = image_shape[:2]
    scale = np.array([w, h, w, h])
    boxes = boxes * scale

    # Perform Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold
    )
    filtered_boxes = []
    for i in indices:
        idx = i[0]
        filtered_boxes.append([*boxes[idx], scores[idx], class_ids[idx]])

    return filtered_boxes

# Read and preprocess the image
image = cv2.imread("input/under_roast (1)_mp4-0013.jpg")
input_tensor = preprocess(image)

# Run inference
outputs = session.run([output_name], {input_name: input_tensor})

# Postprocess the outputs
boxes = postprocess(outputs, image.shape)

# Draw the results on the image
for box in boxes:
    x1, y1, x2, y2, conf, cls = box
    label = f"{int(cls)}: {conf:.2f}"
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save the result
cv2.imwrite("output/result.jpg", image)
print("Inference complete. Result saved to 'output/result.jpg'.")
