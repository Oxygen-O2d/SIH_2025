# app.py
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Use a cache to load the model and classes only once
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

# Get model details
if interpreter:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Get the input size from the model
    _, height, width, _ = input_details[0]['shape']

# --- Streamlit App Interface ---
st.title("🐄 Cattle & Buffalo Classifier")
st.write("Upload an image to classify the animal type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and interpreter:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image and predict on button click
    if st.button('Classify Animal'):
        # Resize image to the model's expected input size
        img_resized = image.resize((height, width))
        img_array = np.array(img_resized, dtype=np.float32)
        img_batch = np.expand_dims(img_array, 0)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_batch)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        predicted_class = class_names[np.argmax(prediction[0])]
        confidence = np.max(prediction[0])
        
        st.success(f"**Prediction:** {predicted_class}")
        st.info(f"**Confidence:** {confidence * 100:.2f}%")
