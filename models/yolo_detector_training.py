from ultralytics import YOLO


model = YOLO("yolov8n.pt")

results = model.train(data='./yolo_dataset/data.yaml', epochs=50, imgsz=1024)