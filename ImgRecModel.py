import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# Dataset Path: Replace with your dataset folder
DATASET_PATH = "C:/Users/roope/OneDrive/Desktop/MajorProject/HarmfulInsects"
IMG_SIZE = 224  # Input size for MobileNetV2
BATCH_SIZE = 32
EPOCHS = 30  # Increased epochs for better learning

# Check if the dataset path exists
if not os.path.exists(DATASET_PATH):
    print(f"Dataset folder '{DATASET_PATH}' not found! Please check the path.")
    exit()

# 1. Data Preprocessing and Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# 2. Load Pre-trained MobileNetV2 and Add Custom Layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Freeze the base model layers
base_model.trainable = False

# Add custom layers on top of the pre-trained model
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global pooling to reduce dimensions
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)  # Regularization to prevent overfitting
output = Dense(train_generator.num_classes, activation='softmax')(x)  # Output layer

# Compile the complete model
model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 3. Train the Model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# 4. Fine-Tune the Base Model
print("Fine-tuning the model...")
base_model.trainable = True  # Unfreeze base model layers

# Use a lower learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train again with the base model unfreezed
history_finetune = model.fit(
    train_generator,
    epochs=25,  # Small number of epochs for fine-tuning
    validation_data=val_generator
)

# 5. Save the Improved Model
model.save("improved_insect_model.h5")
print("Model training complete. Model saved as 'improved_insect_model.h5'.")

# 6. Plot Accuracy and Loss
plt.plot(history.history['accuracy'] + history_finetune.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'] + history_finetune.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
pest_detcection_model.h5