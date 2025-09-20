# app.py
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Cattle & Buffalo Classifier",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Model Loading ---
@st.cache_resource
def load_model_and_classes():
    try:
        interpreter = tf.lite.Interpreter(model_path="model_fp16.tflite")
        interpreter.allocate_tensors()
        with open("class_names.txt", "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        return interpreter, class_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

interpreter, class_names = load_model_and_classes()

if interpreter:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    _, height, width, _ = input_details[0]['shape']

# --- Sidebar ---
with st.sidebar:
    st.title("🐃 Cattle & Buffalo Classifier")
    st.header("Upload Your Image")
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed"
    )
    st.info(
        "This app classifies images of cattle and buffaloes using a trained "
        "deep learning model."
    )

# --- Main Page ---
st.header("Animal Classification")

if uploaded_file is None:
    st.info("Please upload an image using the sidebar to get started.")
else:
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Your Image")
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_container_width=True)
    
    with col2:
        st.subheader("Prediction Results")
        if st.button('Classify Animal'):
            with st.spinner('Analyzing the image...'):
                time.sleep(1) # Small delay for better UX
                
                # Preprocess image
                img_resized = image.resize((height, width))
                img_array = np.array(img_resized, dtype=np.float32)
                img_batch = np.expand_dims(img_array, 0)
                
                # Run inference
                interpreter.set_tensor(input_details[0]['index'], img_batch)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])
                
                predicted_class = class_names[np.argmax(prediction[0])]
                confidence = np.max(prediction[0])
                
                # Display results with style
                st.success(f"**Prediction:** {predicted_class.replace('_', ' ').title()}")
                
                st.write("**Confidence Score:**")
                st.progress(float(confidence))
                st.metric(label="Confidence", value=f"{confidence * 100:.2f}%")

        else:
            st.info("Click the button to classify the uploaded image.")
