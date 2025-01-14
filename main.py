import flet as ft
import cv2
import numpy as np
import base64
import onnxruntime as ort

# Load YOLOv8 ONNX model
onnx_session = ort.InferenceSession("model/yolov8n.onnx")
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# Define live view app with Flet
def main(page: ft.Page):
    page.title = "YOLOv8 Image Upload Detection"

    # Image display and detection table
    uploaded_image = ft.Image()
    detection_table = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("Label")),
            ft.DataColumn(ft.Text("Confidence")),
        ],
        rows=[]
    )

    def process_image(file_path):
        # Read and preprocess the image
        image = cv2.imread(file_path)
        h, w, _ = image.shape
        crop_size = 600
        start_x = max((w - crop_size) // 2, 0)
        start_y = max((h - crop_size) // 2, 0)
        cropped_image = image[start_y:start_y + crop_size, start_x:start_x + crop_size]

        input_frame = cv2.resize(cropped_image, (640, 640))
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        input_frame = input_frame.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_frame = np.expand_dims(input_frame, axis=0)

        # YOLOv8 ONNX inference
        outputs = onnx_session.run([output_name], {input_name: input_frame})
        predictions = outputs[0][0]

        # Annotate the image with bounding boxes
        annotated_image = cropped_image.copy()
        detections = []
        for pred in predictions:
            x1, y1, x2, y2, conf, cls = pred[:6]
            if conf > 0.5:
                label = f"Class {int(cls)}"
                confidence = float(conf)
                detections.append((label, confidence))
                cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(annotated_image, f"{label}: {confidence:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Encode the image as JPEG
        _, buffer = cv2.imencode('.jpg', annotated_image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        # Update UI elements
        uploaded_image.src_base64 = encoded_image
        detection_table.rows = [
            ft.DataRow(cells=[
                ft.DataCell(ft.Text(label)),
                ft.DataCell(ft.Text(f"{confidence:.2f}"))
            ]) for label, confidence in detections
        ]
        page.update()

    def on_upload(event):
        if event.files:
            file_path = event.files[0].path
            process_image(file_path)

    # File upload button
    upload_button = ft.FilePicker(on_result=on_upload)
    page.overlay.append(upload_button)

    # Set up page layout
    page.add(
        ft.Row([
            ft.Column([
                ft.ElevatedButton("Upload Image", on_click=lambda _: upload_button.pick_files(allow_multiple=False)),
                uploaded_image
            ], expand=2),
            ft.Column([
                detection_table
            ], expand=1)
        ])
    )

# Run the app
ft.app(target=main)
