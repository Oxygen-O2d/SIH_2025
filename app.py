# app.py
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Pashu Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- Custom CSS for a Cleaner Look ---
st.markdown("""
    <style>
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 5rem;
            padding-right: 5rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- MODIFIED SECTION: Load the Keras Model ---
@st.cache_resource
def load_model_and_classes():
    try:
        # Load the full Keras model
        model = tf.keras.models.load_model("preliminary_model.keras")
        with open("class_names.txt", "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        return model, class_names
    except Exception as e:
        st.error(f"Error: Could not load the model. Please ensure 'preliminary_model.keras' and 'class_names.txt' are in the root directory.")
        return None, None

model, class_names = load_model_and_classes()

if model:
    # Get the input size from the model's first layer
    input_shape = model.input_shape
    _, height, width, _ = input_shape

# --- Sidebar ---
with st.sidebar:
    st.title("🐾 **Pashu Classifier**")
    st.write("---")
    st.header("Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose an image of a cattle or buffalo...",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed"
    )
    st.write("---")
    st.info(
        "**About:** This tool uses a deep learning model to classify animal types. "
        "It's currently trained to differentiate between general cattle and buffaloes."
    )

# --- Main Page ---
st.title("Image-based Animal Type Classification")
st.markdown("Upload an image via the sidebar to begin classification.")

if uploaded_file is None:
    st.image("https://placehold.co/1200x600/F0F2F6/333333?text=Upload+an+Image+to+Start&font=inter", use_container_width=True)
else:
    col1, col2 = st.columns([0.6, 0.4], gap="large")
    with col1:
        st.subheader("Image Preview")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True, caption="Your Uploaded Image")

    with col2:
        st.subheader("Classification Results")
        if st.button("▶️ Classify Animal", use_container_width=True, type="primary"):
            with st.spinner("Analyzing..."):
                time.sleep(1)
                img_resized = image.resize((height, width))
                img_array = np.array(img_resized, dtype=np.float32)
                img_batch = np.expand_dims(img_array, 0)

                # --- MODIFIED SECTION: Use Keras model.predict ---
                prediction = model.predict(img_batch)
                
                predicted_class = class_names[np.argmax(prediction[0])]
                confidence = np.max(prediction[0])

                st.markdown("---")
                if predicted_class.lower() == 'cattle':
                    st.success(f"### 🐄 Predicted Animal: **{predicted_class.title()}**")
                else:
                    st.success(f"### 🐃 Predicted Animal: **{predicted_class.title()}**")
                
                st.metric(label="Confidence Score", value=f"{confidence * 100:.2f}%")
                st.progress(float(confidence))
        else:
            st.info("Click the button to see the classification.")
