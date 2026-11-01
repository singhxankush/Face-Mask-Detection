import os

# Dataset location
dataset_path = "dataset"

# Two categories
categories = ["with_mask", "without_mask"]

for category in categories:

    folder_path = os.path.join(dataset_path, category)

    # Get all files
    images = os.listdir(folder_path)

    # Count image files
    image_count = len([
        img for img in images
        if img.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print(f"{category}: {image_count} images")

print("\nDataset verification completed!")