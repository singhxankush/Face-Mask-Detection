import tensorflow as tf

# 1. Dataset settings
dataset_path = "dataset"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

# 2. Load training data
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# 3. Load validation data
val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# 4. Check class labels
class_names = train_ds.class_names

print("Class names:", class_names)

# 5. Optimize dataset loading
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.shuffle(1000).prefetch(
    buffer_size=AUTOTUNE
)

val_ds = val_ds.prefetch(
    buffer_size=AUTOTUNE
)

# 6. Build CNN model
model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(128, 128, 3)),

    # Normalize pixels
    tf.keras.layers.Rescaling(1./255),

    # First convolution block
    tf.keras.layers.Conv2D(
        32, (3, 3), activation='relu'
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Second convolution block
    tf.keras.layers.Conv2D(
        64, (3, 3), activation='relu'
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Third convolution block
    tf.keras.layers.Conv2D(
        128, (3, 3), activation='relu'
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Classification layers
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128, activation='relu'
    ),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(
        2, activation='softmax'
    )
])

# 7. Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 8. Display model structure
model.summary()

# 9. Train model
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stop]
)

# 10. Save trained model
model.save("mask_detector.keras")

print("Model training completed!")