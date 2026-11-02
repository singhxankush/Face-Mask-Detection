#  AI Face Mask Detection

A beginner-friendly computer vision and deep learning project that detects faces in images and classifies them into two categories: **With Mask** and **Without Mask**.

Built using Python, TensorFlow, OpenCV and Streamlit.

##  Features

- Upload JPG, JPEG and PNG images.
- Detect multiple faces in a single photograph.
- Classify each detected face using a CNN model.
- Display green bounding boxes for With Mask predictions.
- Display red bounding boxes for Without Mask predictions.
- Show individual prediction confidence scores.
- Display a summary of detected faces.
- Download the annotated result image.
- Reset the application and test another image.

##  Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow / Keras | CNN model training and prediction |
| OpenCV | Face detection using Haar Cascade |
| NumPy | Image array processing |
| Pillow | Image manipulation and annotations |
| Streamlit | Web application interface |
| Matplotlib | Dataset visualization |
| Scikit-learn | Model evaluation and classification report |

##  Dataset

The dataset contains two categories:

| Category | Number of images |
|---|---:|
| With Mask | 3,725 |
| Without Mask | 3,828 |
| **Total** | **7,553** |

The dataset is organized into two folders:

```text
dataset/
├── with_mask/
└── without_mask/
```

##  Model Architecture

A Convolutional Neural Network (CNN) is used for mask classification.

The model contains:

1. Input layer: 128 × 128 RGB image.
2. Rescaling layer: Normalizes pixel values.
3. Convolutional layer: 32 filters.
4. Max pooling layer.
5. Convolutional layer: 64 filters.
6. Max pooling layer.
7. Convolutional layer: 128 filters.
8. Max pooling layer.
9. Flatten layer.
10. Dense layer: 128 neurons with ReLU activation.
11. Dropout layer: 0.5.
12. Output layer: 2 neurons with Softmax activation.

The model is trained using the Adam optimizer and Sparse Categorical Crossentropy loss function.

##  How It Works

The application follows this workflow:

```text
Upload Image
     |
     v
OpenCV Face Detection
     |
     v
Crop Individual Faces
     |
     v
Resize Faces to 128 × 128
     |
     v
CNN Classification
     |
     v
With Mask / Without Mask
     |
     v
Display Results and Bounding Boxes
```

OpenCV locates faces using a Haar Cascade classifier. The CNN then classifies each detected face independently.

##  Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/singhxankush/Face-Mask-Detection.git
cd Face-Mask-Detection
```



### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Place the dataset in the following structure:

```text
dataset/
├── with_mask/
└── without_mask/
```

### 5. Train the model

```bash
python train.py
```

This generates the trained model:

```text
mask_detector.keras
```

If you already have the trained model, you can skip this step.

### 6. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

##  Model Evaluation

The recorded validation evaluation achieved approximately **95.63% accuracy**.

This is a validation result, not a guarantee of accuracy on unfamiliar real-world photographs. Performance can vary with lighting, camera angles, face size and image quality.

##  Limitations

- Haar Cascade may fail to detect some faces, especially small, rotated or partially obscured faces.
- CNN predictions may be incorrect on photographs that differ from the training dataset.
- Confidence scores are model outputs and should not be interpreted as guaranteed accuracy.
- The application processes uploaded images; it does not perform continuous real-time webcam detection.

##  Future Improvements

- Improve face detection for different angles and lighting conditions.
- Train and evaluate the CNN on more diverse images.
- Explore more advanced face detection methods.
- Add real-time webcam detection.

##  Author

**Ankush Singh**

Computer Science and Engineering Student


