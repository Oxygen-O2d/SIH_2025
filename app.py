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
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a Polished Look ---
st.markdown("""
    <style>
        /* Main page styling */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }
        /* Hide Streamlit's default header and footer */
        header {visibility: hidden;}
        footer {visibility: hidden;}
        /* Style the sidebar */
        .st-emotion-cache-16txtl3 {
            padding: 2rem 1.5rem;
        }
        /* Style the primary button */
        .stButton>button {
            border-radius: 20px;
            border: 1px solid #00aaff;
            background-color: #00aaff;
            color: white;
            transition: all 0.2s ease-in-out;
        }
        .stButton>button:hover {
            border-color: #0088cc;
            background-color: #0088cc;
        }
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

# --- Sidebar Content ---
with st.sidebar:
    st.title("🐄 **Pashu Drishti**")
    st.write("---")
    st.header("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image of a cattle or buffalo...",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed"
    )
    st.write("---")
    st.info(
        "**About:** This tool uses a deep learning model to classify animal types. "
        "It's currently trained to differentiate between general cattle and buffaloes. "
        "Perfect for a preliminary check in areas like Gujarat, home of the Gir cow!"
    )

# --- Main Page Content ---
st.title("Image-based Animal Type Classification")

if model is None:
    st.warning("Model is not loaded. Please check the logs for errors.")
else:
    input_shape = model.input_shape
    _, height, width, _ = input_shape

    if uploaded_file is None:
        st.info("Please upload an image using the sidebar to begin.")
        st.image("https://placehold.co/1200x600/F0F2F6/333333?text=Your+Image+Here&font=inter", use_container_width=True)
    else:
        # Layout with columns
        col1, col2 = st.columns([0.6, 0.4], gap="large")

        with col1:
            st.subheader("Image Preview")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True, caption="Your Uploaded Image")

        with col2:
            st.subheader("Classification Results")
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
                    st.success(f"### {emoji} Predicted Animal: **{predicted_class.title()}**")
                    st.metric(label="Confidence Score", value=f"{confidence * 100:.2f}%")
                    st.progress(float(confidence))
            else:
                st.info("Click the button to see the classification.")
