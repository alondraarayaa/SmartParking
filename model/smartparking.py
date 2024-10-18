import cv2
from ultralytics import YOLO,solutions

# Video capture
cap = cv2.VideoCapture("parking_1920_1080.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("parking management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize parking management object
parking_manager = solutions.ParkingManagement(
    model="yolov8n.pt",  # ruta al archivo del modelo
    json_file="bounding_boxes.json",  # ruta al archivo de anotaciones de estacionamiento
)

while cap.isOpened():
    ret, im0 = cap.read()
    if not ret:
        break
    im1 = parking_manager.display_frames(im0)
    video_writer.write(im1)

cap.release()
video_writer.release()
cv2.destroyAllWindows()