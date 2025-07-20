import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image

# Load the trained model
model = load_model('mnist_cnn_model.h5')
st.write("Model loaded successfully!")

# Function to preprocess the drawn image
def preprocess_image(image):
    # Convert to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize to 28x28
    image = cv2.resize(image, (28, 28))
    
    # Normalize and invert colors (MNIST expects white background)
    image = image.astype('float32') / 255.0
    image = 1.0 - image  # Invert colors (black background to white)
    
    # Add channel dimension
    image = np.expand_dims(image, axis=-1)
    
    return image

# Function to make predictions
def predict(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(np.array([processed_image]))
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class] * 100
    
    # Create a dictionary with class probabilities
    probabilities = {str(i): float(prediction[0][i]) for i in range(10)}
    
    return predicted_class, confidence, probabilities

# Streamlit app
st.set_page_config(page_title="Handwritten Digit Recognition", layout="wide")

# Title and description
st.title("🎨 Handwritten Digit Recognition")
st.write("Draw a digit (0-9) in the canvas below and click 'Predict Digit'!")

# Create a canvas component
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Drawing Canvas")
    
    # Create a canvas with drawing capabilities
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",  # White background
        stroke_width=15,
        stroke_color="#000000",  # Black drawing
        background_color="#FFFFFF",  # White background
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    # Buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        predict_btn = st.button("🚀 Predict Digit", use_container_width=True)
    with col_btn2:
        clear_btn = st.button("🧹 Clear Canvas", use_container_width=True)

# Prediction results
with col2:
    st.subheader("Prediction Results")
    
    # Placeholders for results
    prediction_placeholder = st.empty()
    confidence_placeholder = st.empty()
    plot_placeholder = st.empty()
    
    # If the clear button is pressed, clear the canvas
    if clear_btn:
        st.experimental_rerun()
    
    # If the predict button is pressed, make prediction
    if predict_btn and canvas_result.image_data is not None:
        # Convert the canvas image to numpy array
        image = np.array(canvas_result.image_data)
        # The canvas returns an RGBA image, we take only RGB (ignore alpha)
        image = image[:, :, :3]
        
        # Make prediction
        predicted_class, confidence, probabilities = predict(image)
        
        # Display results
        prediction_placeholder.success(f"**Predicted Digit:** {predicted_class}")
        confidence_placeholder.info(f"**Confidence:** {confidence:.1f}%")
        
        # Plot probabilities
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(probabilities.keys(), probabilities.values())
        ax.set_xlabel('Digit')
        ax.set_ylabel('Probability')
        ax.set_title('Prediction Probabilities')
        ax.set_ylim(0, 1)
        plot_placeholder.pyplot(fig)
    else:
        prediction_placeholder.info("Draw a digit and click 'Predict Digit'")
        confidence_placeholder.empty()
        plot_placeholder.empty()

# Examples section
st.subheader("Examples")
st.write("Click any example below to see predictions")

# Create example images directory if not exists
os.makedirs("examples", exist_ok=True)

# Generate example images
example_cols = st.columns(5)
for i in range(10):
    path = f"examples/{i}.png"
    
    # Create image if it doesn't exist
    if not os.path.exists(path):
        # Create a simple image with the digit
        img = np.zeros((28, 28), dtype=np.uint8)
        img = cv2.putText(
            img, str(i), (14, 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2
        )
        cv2.imwrite(path, img)
    
    # Display example in columns
    with example_cols[i % 5]:
        # Fixed the deprecated parameter
        st.image(path, caption=f"Digit {i}", use_container_width=True)
        
        # Use a unique key for each button
        if st.button(f"Predict Example {i}", key=f"example_{i}"):
            # Load the example image
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
            # Make prediction
            predicted_class, confidence, probabilities = predict(image)
            
            # Display results
            prediction_placeholder.success(f"**Predicted Digit:** {predicted_class}")
            confidence_placeholder.info(f"**Confidence:** {confidence:.1f}%")
            
            # Plot probabilities
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(probabilities.keys(), probabilities.values())
            ax.set_xlabel('Digit')
            ax.set_ylabel('Probability')
            ax.set_title('Prediction Probabilities')
            ax.set_ylim(0, 1)
            plot_placeholder.pyplot(fig)

# About section
st.subheader("About this App")
st.markdown("""
- **Model Architecture**: 3 Convolutional layers with MaxPooling and Dropout
- **Training Data**: 60,000 MNIST handwritten digits
- **Accuracy**: >98% on test set

**How to Use:**
1. Draw a digit in the canvas
2. Click 'Predict Digit' to see results
3. Click 'Clear Canvas' to start over
4. Try the example digits for quick testing
""")