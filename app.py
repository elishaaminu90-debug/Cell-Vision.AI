import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# load the trained model
model = load_model('malaria_cnn_model.h5')

st.title('MalariaScan 🔬')
st.write('Upload a blood cell image to check for signs of malaria infection')

uploaded_file = st.file_uploader('Choose a cell image...', type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', width=300)

    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button('Predict'):
        prediction = model.predict(img_array)[0][0]

        if prediction > 0.5:
            confidence = prediction
            st.success('✅ Result: Uninfected')
            st.write(f'Confidence: {confidence:.2%}')
        else:
            confidence = 1 - prediction
            st.error('🦠 Result: Parasitized (Infected)')
            st.write(f'Confidence: {confidence:.2%}')
            st.write('⚠️ Please consult a medical professional for confirmation.')
