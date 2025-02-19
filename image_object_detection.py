import cv2
from yolov8 import YOLOv8

# Initialize yolov8 object detector
model_path = "models/best640.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3)

# Path to your local image
img_path = "image/shell (1)_mp4-0088.jpg"  # <- Update this path

# Read image using OpenCV
img = cv2.imread(img_path)

# Check if image was loaded successfully
if img is None:
    print(f"Error: Could not load image from {img_path}")
    exit()

# Resize the image to 640x640
resized_img = cv2.resize(img, (640, 640))

# Detect Objects
boxes, scores, class_ids = yolov8_detector(resized_img)

# Draw detections
combined_img = yolov8_detector.draw_detections(resized_img)
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Objects", combined_img)
cv2.imwrite("detected_objects.jpg", combined_img)  # Save result
cv2.waitKey(0)