import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model("digit.h5")  # Replace with your model path

# Function to preprocess user-uploaded images
def preprocess_image(image):
    image = tf.image.decode_image(image, channels=1)
    image = tf.image.resize(image, (28, 28))
    image = tf.cast(image, tf.float32) / 255.0
    return image

# Streamlit UI
st.title("Digit Recognizer")

# Sidebar for image upload
st.sidebar.title("Upload Image")
uploaded_image = st.sidebar.file_uploader("Upload a digit image for recognition", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = preprocess_image(uploaded_image.read())
    st.sidebar.image(image, caption="Uploaded Image", use_column_width=True)
    image = np.expand_dims(image, axis=0)

    # Make predictions
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    st.write("### Prediction:")
    st.write(f"The uploaded image is predicted to be digit: {predicted_class}")

# Display sample images and their predictions
st.write("### Sample Images and Predictions")
num_samples = 10
sample_indices = np.random.choice(len(X_val), num_samples, replace=False)
sample_images = X_val[sample_indices]
sample_labels = y_val.iloc[sample_indices]
sample_predictions = model.predict(sample_images)

fig, axes = plt.subplots(2, 5, figsize=(12, 8))
for i in range(num_samples):
    row, col = i // 5, i % 5
    axes[row, col].imshow(sample_images[i][:, :, 0], cmap="gray")
    axes[row, col].set_title(f"Predicted: {np.argmax(sample_predictions[i])}\nActual: {sample_labels.iloc[i]}")
    axes[row, col].axis("off")

st.pyplot(fig)

# Optionally, display model summary
st.write("### Model Summary")
st.text(model.summary())
