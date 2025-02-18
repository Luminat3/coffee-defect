import onnxruntime as ort
import numpy as np
import cv2

# Load the ONNX model
session = ort.InferenceSession("model/best640.onnx")

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
    predictions = outputs[0]  # Extract the first (and only) batch
    
    # Check if the output shape matches the expected format
    if predictions.shape[1] != 10:
        raise ValueError(f"Unexpected output shape: {predictions.shape}")
    
    predictions = predictions.squeeze(0)  # Remove batch dimension: [10, 8400]
    
    # Extract components
    boxes = predictions[:4, :].T  # Bounding box coordinates: [8400, 4]
    object_conf = predictions[4, :]  # Object confidence scores: [8400]
    class_probs = predictions[5:, :].T  # Class probabilities: [8400, num_classes]
    class_ids = np.argmax(class_probs, axis=1)  # Class IDs: [8400]
    class_conf = np.max(class_probs, axis=1)  # Class confidence scores: [8400]

    # Combine object confidence with class confidence
    scores = object_conf * class_conf
    
    # Filter by confidence threshold
    mask = scores > conf_threshold
    if mask.sum() == 0:
        return []  # No valid detections
    
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    
    # Rescale boxes to the original image size
    h, w = image_shape[:2]
    scale = np.array([w, h, w, h])
    boxes = boxes * scale
    
    # Perform Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold
    )
    
    filtered_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            filtered_boxes.append([*boxes[i], scores[i], class_ids[i]])
    
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