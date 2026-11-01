
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

from PIL import Image, ImageDraw

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Face Mask Detection",
    page_icon="😷",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mask_detector.keras")

model = load_model()

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

class_names = ["With Mask", "Without Mask"]

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
    }

    .main-title {
        font-size: 34px;
        font-weight: 700;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .result-card {
        background-color: #1e293b;
        padding: 18px;
        border-radius: 12px;
        text-align: center;
    }

    .result-number {
        font-size: 30px;
        font-weight: bold;
    }

    .result-label {
        color: #cbd5e1;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown(
    '<div class="main-title">😷 AI Face Mask Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Computer Vision • OpenCV • CNN</div>',
    unsafe_allow_html=True
)

st.divider()

# ---------------- UPLOAD ----------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    # Two-column layout
    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Uploaded Image")

        st.image(
            image,
            width=450
        )

    with right:
        st.subheader("Instructions")

        st.write(
            "Upload a clear image containing one or more faces."
        )

        st.write(
            "Our application detects faces and classifies "
            "each detected face."
        )

        detect_button = st.button(
            "🔍 Detect Faces",
            type="primary",
            use_container_width=True
        )

    # ---------------- DETECTION ----------------

    if detect_button:

        gray = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2GRAY
        )

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        result_image = image.copy()
        draw = ImageDraw.Draw(result_image)

        with_mask = 0
        without_mask = 0
        results = []

        for i, (x, y, w, h) in enumerate(faces):

            face = image_array[y:y+h, x:x+w]

            face = Image.fromarray(face).resize((128, 128))
            face_array = np.array(face)
            face_array = np.expand_dims(face_array, axis=0)

            predictions = model.predict(
                face_array, verbose=0
            )[0]

            predicted_index = np.argmax(predictions)
            predicted_class = class_names[predicted_index]

            confidence = float(
                predictions[predicted_index]
            ) * 100

            if predicted_index == 0:
                with_mask += 1
                color = (0, 200, 0)
            else:
                without_mask += 1
                color = (255, 50, 50)

            # Bounding box
            draw.rectangle(
                (x, y, x+w, y+h),
                outline=color,
                width=4
            )

            # Label
            label = f"{predicted_class} {confidence:.1f}%"

            text_box = draw.textbbox((0, 0), label)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]

            label_y = max(0, y - text_height - 10)

            draw.rectangle(
                (x, label_y, x + text_width + 10,
                 label_y + text_height + 8),
                fill=color
            )

            draw.text(
                (x + 5, label_y + 3),
                label,
                fill=(255, 255, 255)
            )

            results.append(
                (i+1, predicted_class, confidence)
            )

        # ---------------- RESULTS ----------------

        st.divider()
        st.subheader("📊 Detection Summary")

        if len(faces) == 0:

            st.warning("No faces detected. Try another image.")

        else:

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Faces", len(faces))
            col2.metric("With Mask", with_mask)
            col3.metric("Without Mask", without_mask)

            st.divider()

            result_col, details_col = st.columns([1.2, 1])

            with result_col:
                st.subheader("Detection Results")

                st.image(
                    result_image,
                    width=500
                )

            with details_col:
                st.subheader("Individual Predictions")

                for face_id, prediction, confidence in results:

                    if prediction == "With Mask":
                        st.success(
                            f"Face {face_id}: {prediction}"
                        )
                    else:
                        st.error(
                            f"Face {face_id}: {prediction}"
                        )

                    st.caption(
                        f"Confidence: {confidence:.2f}%"
                    )

            st.caption(
                "Note: Predictions are model estimates and "
                "may not always be correct."
            )
