import cv2
from ultralytics import solutions

polygon_json_path = "bounding_boxes_crop.json"
cap = cv2.VideoCapture("parking_crop.mp4")
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("parking management.avi", 
cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

management = solutions.ParkingManagement(model="yolov8n.pt", json_file="bounding_boxes.json") 

while cap.isOpened():
  ret, im0 = cap.read()
  if not ret:
    break
   
results = management.model.track(im0, persist=True, show=False)

if results[0].boxes.id is not None:
  boxes = results[0].boxes.xyxy.cpu().tolist()
  clss = results[0].boxes.cls.cpu().tolist()
  management.process_data(json_file, im0, boxes, clss)

management.display_frames(im0)
video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()