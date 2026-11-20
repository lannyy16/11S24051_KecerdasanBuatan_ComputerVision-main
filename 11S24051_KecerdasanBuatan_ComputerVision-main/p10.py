from ultralytics import YOLO

model = YOLO("runs/detect/runs/latihan/coba/weights/best.pt")
model.predict(source="video.mp4", save=True)
