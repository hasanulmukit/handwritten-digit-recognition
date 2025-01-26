import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('digit_recognition_model.h5')

# App configuration
st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✍️")

# App title and description
st.title("Handwritten Digit Recognition ✍️")
st.markdown("""
    Upload an image of a digit (0-9) written on a white background, or draw a digit on the canvas below to predict it using a trained deep learning model.
""")

# Sidebar options for canvas
st.sidebar.header("Canvas Options")
stroke_width = st.sidebar.slider("Stroke Width: ", 1, 25, 9)
stroke_color = "#ffffff"  # Black stroke color
bg_color = "#000000"  # White background
realtime_update = st.sidebar.checkbox("Update in Real Time", True)

# Drawing canvas
st.markdown("### Draw a digit:")
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # No fill color
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    update_streamlit=realtime_update,
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Preprocessing function
def preprocess_image(image):
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28)).convert('L')  # Resize and convert to grayscale
    image_array = np.array(image).astype('float32') / 255  # Normalize to [0, 1]
    image_array = image_array.reshape(1, 28, 28, 1)  # Add batch and channel dimensions
    return image_array

# Predict from canvas
if canvas_result.image_data is not None:
    canvas_image = canvas_result.image_data
    st.image(canvas_image, caption="Canvas Image", use_container_width=False)

    # Predict when button is clicked
    if st.button("Predict from Canvas"):
        canvas_image_pil = Image.fromarray((canvas_image[:, :, :3] * 255).astype('uint8')).convert('L')
        processed_image = preprocess_image(canvas_image_pil)
        prediction = model.predict(processed_image)
        predicted_digit = np.argmax(prediction)
        confidence = prediction[0][predicted_digit] * 100

        st.success(f"Predicted Digit: {predicted_digit}")
        st.info(f"Confidence: {confidence:.2f}%")

# File upload for digit images
uploaded_file = st.file_uploader("Or upload a digit image:", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)
    confidence = prediction[0][predicted_digit] * 100

    # Display results
    st.success(f"Predicted Digit: {predicted_digit}")
    st.info(f"Confidence: {confidence:.2f}%")

# Footer
st.markdown("""
    <style>
        footer {visibility: hidden;}
        .footer {
            text-align: center;
            font-size: 12px;
            color: gray;
            margin-top: 20px;
        }
    </style>
    <div class="footer">
        Built with ❤️ by Hasanul Mukit
    </div>
""", unsafe_allow_html=True)
