from ultralytics import YOLO

model = YOLO("yolo26n.pt")                    # YOLO detector
results = model.predict(source="sample.jpg",  # kamera / foto
                        conf=0.25,            # filter confidence
                        save=True)

target = "person"                             # filter class
count = 0                                     # counting logic
for result in results:
    for box in result.boxes:
        if model.names[int(box.cls[0])] == target:
            count += 1

print(f"Jumlah {target}: {count}")            # tampilkan hasilfrom ultralytics import YOLO

model = YOLO("yolo26n.pt")                    # YOLO detector
results = model.predict(source="sample.jpg",  # kamera / foto
                        conf=0.25,            # filter confidence
                        save=True)

target = "person"                             # filter class
count = 0                                     # counting logic
for result in results:
    for box in result.boxes:
        if model.names[int(box.cls[0])] == target:
            count += 1

print(f"Jumlah {target}: {count}")            # tampilkan hasil