import flet as ft
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import base64
from io import BytesIO


def pil_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def resize_with_aspect_ratio(image, size):
    h, w, _ = image.shape
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    delta_w, delta_h = size - new_w, size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded_image

def predict(image):
    session = ort.InferenceSession("model/yolov8n.onnx")
    input_name = session.get_inputs()[0].name
    image_data = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
    image_data = np.expand_dims(image_data, axis=0)
    predictions = session.run(None, {input_name: image_data})
    return predictions

def main(page: ft.Page):
    page.title = "YOLOv8 Image Prediction"

    def process_image(e):
        if not file_picker.result or not file_picker.result.files:
            return

        # Ambil path file dari file pertama yang dipilih
        img_path = file_picker.result.files[0].path
        image = cv2.imread(img_path)
        resized_image = resize_with_aspect_ratio(image, 640)
        predictions = predict(resized_image)

        # Convert resized image back to PIL format for display
        display_image = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        img_control.src_base64 = f"data:image/png;base64,{pil_to_base64(display_image)}"    

        result_text.value = f"Predictions: {predictions}"
        page.update()


    file_picker = ft.FilePicker(on_result=process_image)
    page.overlay.append(file_picker)

    upload_button = ft.ElevatedButton("Upload Image", on_click=lambda _: file_picker.pick_files())
    img_control = ft.Image(width=640, height=640, fit=ft.ImageFit.CONTAIN)
    result_text = ft.Text("Predictions will appear here.")

    page.add(
        ft.Column([
            upload_button,
            img_control,
            result_text
        ])
    )

ft.app(target=main)
