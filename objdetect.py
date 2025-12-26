
import cv2
import numpy as np

# =======================
# 1. Read and display image
# =======================
image_path = "dog.jpg"   # change path if needed
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError("Image not found")

cv2.imshow("Original Image", img)
cv2.waitKey(0)

# =======================
# Convert to grayscale
# =======================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# =======================
# 2. Edge Detection
# =======================
edges = cv2.Canny(gray, 50, 200)
cv2.imshow("Edge Detection", edges)
cv2.waitKey(0)

# =======================
# 3. Thresholding
# =======================
_, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
cv2.imshow("Threshold Image", thresh)
cv2.waitKey(0)

# =======================
# 4. Contour Extraction
# =======================
contours, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

cv2.imshow("Contours", contour_img)
cv2.waitKey(0)

# =======================
# 5. Simple Shape Detection
# =======================
shape_img = img.copy()

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 500:
        continue

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    x, y, w, h = cv2.boundingRect(approx)
    corners = len(approx)

    if corners == 3:
        shape_name = "Triangle"
    elif corners == 4:
        ar = w / float(h)
        shape_name = "Square" if 0.95 <= ar <= 1.05 else "Rectangle"
    elif corners > 4:
        shape_name = "Circle"
    else:
        shape_name = "Unknown"

    cv2.rectangle(shape_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(
        shape_img,
        shape_name,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
    )

cv2.imshow("Shape Detection", shape_img)
cv2.waitKey(0)

# =======================
# 6. Face Detection
# =======================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise IOError("Face cascade XML not found")

face_img = img.copy()
gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray_face, 1.1, 5)

for (x, y, w, h) in faces:
    cv2.rectangle(face_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow("Face Detection", face_img)
cv2.waitKey(0)

cv2.destroyAllWindows()
