
import cv2
from PIL import Image
import numpy as np

# Load OpenCV's pre-trained face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Load an image
image_path = "test_image.jpg"

image = Image.open(image_path).convert("RGB")

# Convert image into NumPy array
image_array = np.array(image)

# Convert RGB image into grayscale
gray = cv2.cvtColor(
    image_array,
    cv2.COLOR_RGB2GRAY
)

# Detect faces
faces = face_detector.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(40, 40)
)

print("Number of faces detected:", len(faces))

# Draw rectangles around detected faces
for (x, y, w, h) in faces:

    cv2.rectangle(
        image_array,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        3
    )

# Display result
result = Image.fromarray(image_array)
result.show()
