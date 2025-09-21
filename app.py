# app.py
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Pashu Drishti | Animal Classifier",
    page_icon="🐄",
    layout="wide",
)

# --- Custom CSS for a Polished, Centered Look ---
st.markdown("""
    <style>
        /* Hide Streamlit's default header and footer */
        header {visibility: hidden;}
        footer {visibility: hidden;}
        /* Center the main content */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 5rem;
            padding-right: 5rem;
            max-width: 1200px;
            margin: auto;
        }
        /* Style the title */
        h1 {
            text-align: center;
            color: #333;
        }
        /* REMOVED the .stFileUploader style to get rid of the white box */
    </style>
""", unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_model_and_classes():
    try:
        model = tf.keras.models.load_model("preliminary_model.keras")
        with open("class_names.txt", "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        return model, class_names
    except Exception as e:
        st.error(f"Error: Could not load the model. Please ensure 'preliminary_model.keras' and 'class_names.txt' are in the root directory.")
        return None, None

model, class_names = load_model_and_classes()

# --- App Title and Header ---
st.title("🐾 Pashu Drishti")
st.markdown("<p style='text-align: center; color: grey;'>An AI-Powered Cattle & Buffalo Classifier</p>", unsafe_allow_html=True)
st.write("---")

# --- File Uploader ---
if model:
    input_shape = model.input_shape
    _, height, width, _ = input_shape

    st.subheader("Upload Your Image Here")
    uploaded_file = st.file_uploader(
        "Click to browse or drag and drop an image",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed"
    )
else:
    uploaded_file = None
    st.warning("Model is not loaded. Please check the logs for errors.")

st.write("---")

# --- Main Content: Image and Results ---
if uploaded_file is None:
    st.info("Please upload an image to get started.")
else:
    col1, col2 = st.columns([0.6, 0.4], gap="large")

    with col1:
        st.subheader("Image Preview")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True, caption="Your Uploaded Image")

    with col2:
        st.subheader("Classification")
        st.write("Click the button below to classify the animal.")
        
        if st.button("▶️ Classify Animal", use_container_width=True, type="primary"):
            with st.spinner("Analyzing..."):
                time.sleep(1) # Simulate processing for better UX

                # Preprocess and predict
                img_resized = image.resize((height, width))
                img_array = np.array(img_resized, dtype=np.float32)
                img_batch = np.expand_dims(img_array, 0)

                prediction = model.predict(img_batch)
                predicted_class = class_names[np.argmax(prediction[0])]
                confidence = np.max(prediction[0])

                # Display results
                st.markdown("---")
                emoji = "🐄" if predicted_class.lower() == 'cattle' else "🐃"
                st.success(f"### {emoji} Prediction: **{predicted_class.title()}**")
                
                st.metric(label="Confidence Score", value=f"{confidence * 100:.2f}%")
                st.progress(float(confidence))

                st.info(f"This tool is great for a first look, especially here in Gujarat where the Gir cow is so common!")
