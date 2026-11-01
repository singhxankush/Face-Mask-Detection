import os
import matplotlib.pyplot as plt
from PIL import Image

dataset_path = "dataset"

categories = ["with_mask", "without_mask"]

plt.figure(figsize=(10, 5))

for i, category in enumerate(categories):

    folder_path = os.path.join(dataset_path, category)

    # Get image filenames
    images = os.listdir(folder_path)

    # Select first image
    image_path = os.path.join(folder_path, images[0])

    # Open image
    img = Image.open(image_path)

    # Display image
    plt.subplot(1, 2, i + 1)
    plt.imshow(img)
    plt.title(category)
    plt.axis("off")

plt.tight_layout()
plt.show()