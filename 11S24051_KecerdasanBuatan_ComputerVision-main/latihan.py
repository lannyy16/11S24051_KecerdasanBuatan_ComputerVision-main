from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.train(data="coco8.yaml", epochs=3, imgsz=640, batch=4, project="runs/latihan", name="coba")