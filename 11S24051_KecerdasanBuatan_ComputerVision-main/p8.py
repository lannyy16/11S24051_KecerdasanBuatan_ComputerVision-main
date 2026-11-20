from ultralytics import YOLO

model = YOLO("runs/detect/runs/latihan/coba/weights/best.pt")

# Evaluasi pada validation split
metrics = model.val(data="coco8.yaml")

# Prediksi gambar
model.predict(source="sample.jpg", save=True)

# Ekspor untuk deployment
model.export(format="onnx")