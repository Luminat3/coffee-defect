import flet as ft
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import io
import base64

# Load ONNX model
model_path = "model/yolov8n.onnx"
session = ort.InferenceSession(model_path)

# Preprocess image
def preprocess_image(image, target_size):
    original_size = image.shape[:2]  # (height, width)

    # Resize while maintaining aspect ratio
    scale = min(target_size / original_size[1], target_size / original_size[0])
    new_size = (int(original_size[1] * scale), int(original_size[0] * scale))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

    # Pad to target size
    padded_image = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    top_left = ((target_size - new_size[1]) // 2, (target_size - new_size[0]) // 2)
    padded_image[top_left[0]:top_left[0] + new_size[1], top_left[1]:top_left[1] + new_size[0]] = resized_image

    # Normalize and transpose to CHW format
    image = padded_image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))[np.newaxis, ...]
    return image, original_size, scale, top_left

# Postprocess outputs
def postprocess_detections(detections, original_size, scale, top_left, conf_threshold=0.25):
    boxes, scores, class_ids = [], [], []
    for detection in detections:
        conf = detection[4]
        if conf > conf_threshold:
            x_center, y_center, width, height = detection[:4]
            x_center = (x_center - top_left[1]) / scale
            y_center = (y_center - top_left[0]) / scale
            width /= scale
            height /= scale
            x1, y1 = x_center - width / 2, y_center - height / 2
            x2, y2 = x_center + width / 2, y_center + height / 2
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
            scores.append(float(conf))
            class_ids.append(int(detection[5]))
    return boxes, scores, class_ids

# Visualize detections
def draw_detections(image, boxes, scores, class_ids, class_names):
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        label = f"{class_names[class_id]}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# GUI with Flet
def main(page: ft.Page):
    page.title = "YOLOv8 Detection"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.scroll = "adaptive"

    # Elements
    result_image = ft.Image(width=640, height=640, fit=ft.ImageFit.CONTAIN)
    file_picker = ft.FilePicker(on_result=lambda e: on_file_upload(e))

    def on_file_upload(e: ft.FilePickerResultEvent):
        if e.files:
            # Read image
            file_path = e.files[0].path
            image = cv2.imread(file_path)

            # Process image
            input_image, original_size, scale, top_left = preprocess_image(image, 640)
            inputs = {session.get_inputs()[0].name: input_image}
            outputs = session.run(None, inputs)

            # Extract and process detections
            detections = outputs[0][0]
            boxes, scores, class_ids = postprocess_detections(detections, original_size, scale, top_left)

            # Draw detections
            annotated_image = draw_detections(image.copy(), boxes, scores, class_ids, [f"class_{i}" for i in range(80)])
            
            # Convert to displayable format
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(annotated_image)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            buffer.seek(0)
            encoded_image = base64.b64encode(buffer.read()).decode("utf-8")
            result_image.src_base64 = f"data:image/png;base64,{encoded_image}"
            page.update()

    # Layout
    page.overlay.append(file_picker)
    page.add(
        ft.Text("Upload an image for YOLOv8 detection", size=20),
        ft.ElevatedButton("Upload Image", on_click=lambda _: file_picker.pick_files()),
        result_image
    )

# Run Flet app
ft.app(target=main)
