import cv2
import matplotlib.pyplot as plt

# Baca citra dari file
image_bgr = cv2.imread("bus.jpg")
if image_bgr is None:
    raise FileNotFoundError("bus.jpg tidak ditemukan")

# OpenCV membaca dalam urutan BGR
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

print("Ukuran (H, W, C):", image_rgb.shape)
plt.imshow(image_rgb)
plt.axis("off")
plt.show()