from ultralytics import YOLO

model = YOLO("runs/detect/runs/latihan/coba/weights/best.pt")
model.predict(source=0, show=True)