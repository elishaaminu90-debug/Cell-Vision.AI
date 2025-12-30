import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

# ========================================================================
# !!! FINAL SOLUTION !!!
# These values are confirmed by the model summary to resolve the 12544 error.
# ========================================================================
IMAGE_HEIGHT = 64  # <--- CORRECTED FIX
IMAGE_WIDTH = 64   # <--- CORRECTED FIX
TARGET_CHANNELS = 3 


# --- Streamlit Setup ---
st.set_page_config(page_title="CellVision AI App", layout="centered")
st.title(' Cell Vision AI Classifier')
st.markdown("Upload a cell image to classify the cell type.")
st.markdown("---")


# --- 1. Efficient Model Loading (Using @st.cache_resource) ---

@st.cache_resource
def load_vision_model():
    """Loads the CellVision_AI.h5 model once using caching."""
    with st.spinner("Loading AI model..."):
        try:
            model = load_model('CellVision_AI.h5')
            return model
        except Exception as e:
            st.error(f"FATAL ERROR: Could not load model 'CellVision_AI.h5'. Ensure the file is in the same folder. Error: {e}")
            st.stop()

cell_vision_model = load_vision_model()


# --- 2. Image Preprocessing Function ---

def preprocess_image(uploaded_file):
    """
    Transforms the uploaded file into a batch tensor ready for prediction, 
    matching the model's required input shape (64x64).
    """
    img = Image.open(uploaded_file).convert('RGB')
    
    # Resizing now uses the CORRECTED (64x64) dimensions
    img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    return img_batch


# --- 3. User Interface and Prediction Logic ---

uploaded_file = st.file_uploader(
    "Upload a cell image (JPG or PNG) for analysis:", 
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    
    col_img, col_result = st.columns([1, 2])
    
    with col_img:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        predict_button = st.button(" Classify Cell")
    
    with col_result:
        st.subheader("Classification Result")
        
        if predict_button:
            with st.spinner("Analyzing image with Cell Vision AI..."):
                
                input_tensor = preprocess_image(uploaded_file)
                
                # Make the prediction (This line caused the error, but should now work!)
                prediction = cell_vision_model.predict(input_tensor)[0]
                
                # --- Post-Processing and Display ---
                predicted_class_index = np.argmax(prediction)
                confidence_score = prediction[predicted_class_index] * 100
                
                # IMPORTANT: Based on the model summary, your model is binary (2 classes).
                # Replace these placeholder labels with your two actual cell types.
                class_labels = ["Type A Cell (e.g., Healthy)", "Type B Cell (e.g., Diseased)"] 
                
                predicted_label = class_labels[predicted_class_index]
                
                st.metric(label="Predicted Cell Type", 
                          value=predicted_label)
                
                st.progress(int(confidence_score), 
                            text=f"Confidence: **{confidence_score:.2f}%**")
        
        else:
            st.info("Upload an image and click 'Classify Cell' to begin the analysis.")
