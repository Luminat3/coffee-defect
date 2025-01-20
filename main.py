import cv2
import numpy as np
import onnxruntime as ort

def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize and pad image while keeping the aspect ratio and ensuring the output is exactly new_shape.
    """
    shape = image.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

    # Compute padding
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw, dh = dw // 2, dh // 2  # divide padding into two sides

    # Resize and pad
    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color
    )

    return padded, (r, r), (dw, dh)

def preprocess(image_path):
    # Load image
    image = cv2.imread(image_path)
    assert image is not None, f"Image not found: {image_path}"

    # Letterbox resize
    img, ratio, dwdh = letterbox(image, new_shape=(640, 640))

    # Convert BGR to RGB and normalize
    img = img[:, :, ::-1].transpose(2, 0, 1)  # HWC to CHW
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0  # normalize to 0-1

    # Ensure the image shape matches model input (N, C, H, W)
    img = np.expand_dims(img, axis=0)  # add batch dimension

    # Validate input dimensions
    assert img.shape == (1, 3, 640, 640), f"Invalid input shape: {img.shape}, expected (1, 3, 640, 640)"

    return img, image, ratio, dwdh

def run_inference(onnx_path, image_path):
    # Load ONNX model
    session = ort.InferenceSession(onnx_path)

    # Preprocess image
    img, original_img, ratio, dwdh = preprocess(image_path)

    # Run inference
    inputs = {session.get_inputs()[0].name: img}
    outputs = session.run(None, inputs)

    return outputs, original_img, ratio, dwdh

def postprocess(outputs, original_img, ratio, dwdh):
    """Postprocess outputs to original image scale."""
    detections = outputs[0]  # assuming single output

    # Rescale boxes to original image size
    detections[:, [0, 2]] -= dwdh[0]  # x padding
    detections[:, [1, 3]] -= dwdh[1]  # y padding
    detections[:, [0, 2]] /= ratio[0]
    detections[:, [1, 3]] /= ratio[1]

    # Clip boxes to image dimensions
    detections[:, [0, 2]] = np.clip(detections[:, [0, 2]], 0, original_img.shape[1])
    detections[:, [1, 3]] = np.clip(detections[:, [1, 3]], 0, original_img.shape[0])

    return detections

# Example usage
onnx_path = 'model/yolov8n.onnx'
image_path = r'input\under_roast (1)_mp4-0013.jpg'  # Replace with your image path

outputs, original_img, ratio, dwdh = run_inference(onnx_path, image_path)
detections = postprocess(outputs, original_img, ratio, dwdh)

# Draw detections on the image
for det in detections:
    if len(det) >= 6:
        x1, y1, x2, y2, conf, cls = det[:6]
        label = f"{int(cls.item())} {conf:.2f}"
        cv2.rectangle(original_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(original_img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imwrite('output.jpg', original_img)
print("Detections saved to output.jpg")
