
import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model

MODEL_PATH = "malaria_cnn_model.h5"
IMAGE_SIZE = (128, 128)
THRESHOLD = 0.5

st.set_page_config(
    page_title="MalariaScan | Cell-Vision.AI",
    page_icon="🔬",
    layout="centered",
)

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH, compile=False)


st.title("🔬 MalariaScan")
st.caption("Cell-Vision.AI | Educational ML demonstration")

st.info(
    "This application demonstrates image classification research. "
    "It is not a validated diagnostic tool and cannot determine "
    "whether a person has malaria."
)

try:
    model = get_model()
except Exception as exc:
    st.error(
        "The model could not be loaded. Check that the model file "
        "exists and that your TensorFlow dependencies are compatible."
    )
    with st.expander("Technical details"):
        st.code(str(exc))
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a microscope cell image",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    try:
        with Image.open(uploaded_file) as source:
            img = source.convert("RGB")

        st.image(img, caption="Uploaded cell image", width=300)

        if st.button("Analyse image", type="primary"):
            with st.spinner("Running model prediction..."):
                resized = img.resize(IMAGE_SIZE)
                img_array = np.asarray(resized, dtype=np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                output = np.asarray(
                    model.predict(img_array, verbose=0)
                ).reshape(-1)

                if output.size != 1 or not np.isfinite(output[0]):
                    st.error("The model returned an unexpected output.")
                    st.stop()

                score = float(output[0])

                if not 0.0 <= score <= 1.0:
                    st.error(
                        "Expected a sigmoid probability-like output "
                        "between 0 and 1. Check the model's output layer."
                    )
                    st.stop()

                # ASSUMPTION: output > threshold means Uninfected.
                # Verify this mapping against the training labels.
                if score > THRESHOLD:
                    predicted_class = "Uninfected"
                    st.success(f"Predicted class: {predicted_class}")
                else:
                    predicted_class = "Parasitized"
                    st.warning(f"Predicted class: {predicted_class}")

                st.metric(
                    "Raw model score",
                    f"{score:.4f}",
                )

                st.caption(
                    "The score is the model output, not a confirmed "
                    "diagnosis or necessarily a calibrated probability."
                )

                st.warning(
                    "Do not use this prediction to make medical "
                    "decisions. Seek assessment from a qualified "
                    "healthcare professional if malaria is suspected."
                )

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"Could not process the uploaded image: {exc}")
    except Exception as exc:
        st.error("An unexpected error occurred while analysing the image.")
        with st.expander("Technical details"):
            st.code(str(exc))
