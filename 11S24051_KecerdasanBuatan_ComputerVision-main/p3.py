import cv2

img = cv2.imread("sample.jpg")
if img is None:
    raise FileNotFoundError("sample.jpg")

resized = cv2.resize(img, (640, 640))
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 80, 160)

cv2.imwrite("resized.jpg", resized)
cv2.imwrite("edges.jpg", edges)

# ===== TAMBAHAN (di luar slide): tampilkan 4 tahap dalam grid =====
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
plt.title("1. Resize 640x640")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(gray, cmap="gray")
plt.title("2. Grayscale")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(blur, cmap="gray")
plt.title("3. Gaussian Blur")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(edges, cmap="gray")
plt.title("4. Canny (tepi)")
plt.axis("off")

plt.tight_layout()
plt.show()