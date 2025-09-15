import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

# --- Real-Time Inference Code ---

# Load the trained model (inference phase)
MODEL_PATH = "improved_insect_model.h5"  # Path to your trained model
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found.")
    exit()

model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# Correct CLASS_NAMES list with all 24 insect classes
CLASS_NAMES = [
    "Africanized Honey Bees (Killer Bees)", "Aphids", "Armyworms", "Beetles", 
    "Brown Marmorated Stink Bugs", "Cabbage Loopers", "Caterpillars", "Citrus Canker", 
    "Colorado Potato Beetles", "Corn Borers", "Corn Earworms", "Earwigs", 
    "Fall Armyworms", "Fruit Flies", "Grasshoppers", "Moths", "Slugs", "Snails", 
    "Spider Mites", "Thrips", "Tomato Hornworms", "Wasps", "Weevils", 
    "Western Corn Rootworms"
]
# Function to preprocess video frames for MobileNetV2
def preprocess_frame(frame, img_size=224):
    """
    Resize, normalize, and prepare the frame for model prediction.
    """
    frame_resized = cv2.resize(frame, (img_size, img_size))  # Resize to model input size
    frame_normalized = frame_resized / 255.0  # Normalize pixel values
    return np.expand_dims(frame_normalized, axis=0)  # Add batch dimension

# Open webcam for video feed
cap = cv2.VideoCapture(0)  # Default webcam
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Press 'q' to quit the program.")
print("Webcam is running. Processing video feed...")

# Real-time video feed processing
while True:
    ret, frame = cap.read()  # Capture each frame
    if not ret:
        print("Error: Could not read the video frame.")
        break

    # Preprocess the current frame
    input_frame = preprocess_frame(frame)

    # Predict using the model
    predictions = model.predict(input_frame, verbose=0)

    # Dynamically check the number of classes
    num_classes = predictions.shape[1]
    if len(CLASS_NAMES) != num_classes:
        print("Error: CLASS_NAMES length does not match the model output.")
        print(f"Expected {num_classes} class names, but got {len(CLASS_NAMES)}.")
        break

    # Retrieve prediction results
    predicted_class = CLASS_NAMES[np.argmax(predictions)]  # Class with the highest probability
    confidence = np.max(predictions)  # Confidence score

    # Display the prediction and confidence score on the frame
    display_text = f"{predicted_class}: {confidence:.2f}"
    cv2.putText(frame, display_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Print the class name and confidence score to the terminal
    print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}", end="\r")

    # Show the frame with prediction overlay
    cv2.imshow("Harmful Insect Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
print("\nProgram terminated successfully.")
