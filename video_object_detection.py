import cv2
from yolov8 import YOLOv8

# Path to the input video
videoPath = "video/under_roast.mp4"

# Open video file
cap = cv2.VideoCapture(videoPath)

# Set start time (skip first {start_time} seconds)
start_time = 5
cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize YOLOv8 model
model_path = "models/best640.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define output video writer
out = cv2.VideoWriter(
    "output.avi", 
    cv2.VideoWriter_fourcc(*"MJPG"), 
    fps, 
    (frame_width, frame_height)
)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

while cap.isOpened():
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Read frame from the video
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Perform object detection
    boxes, scores, class_ids = yolov8_detector(frame)

    # Draw detections on the frame
    combined_img = yolov8_detector.draw_detections(frame)

    # Display results
    cv2.imshow("Detected Objects", combined_img)

    # Write frame to output video
    out.write(combined_img)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
