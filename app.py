import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download


HF_TOKEN = st.secrets["HF_TOKEN"]
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Eyaddddddd/AlzheimerClassifierTL",
        filename="AlzheimerClassifierTL.h5",
        token=HF_TOKEN
    )
    return tf.keras.models.load_model(model_path)

model = load_model()


class_mapping = {
    "MildDemented": 0,
    "ModerateDemented": 1,
    "NonDemented": 2,
    "VeryMildDemented": 3
}
idx_to_class = {v: k for k, v in class_mapping.items()}


status_messages = {
    "NonDemented":  ("✅ Likely Healthy / No Dementia detected", "green"),
    "VeryMildDemented": ("🟡 Signs of Very Mild Dementia", "orange"),
    "MildDemented": ("🟠 Signs of Mild Dementia", "darkorange"),
    "ModerateDemented": ("🔴 Signs of Moderate Dementia", "darkred")
}


def preprocess_image(image):
    IMG_SIZE = (224, 224)
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Detection"])


if page == "Home":
    st.markdown(
        "<h1 style='text-align: center;'>🧠 Alzheimer’s Detection App</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h3 style='text-align: center;'>Transfer Learning with ResNet50</h3>",
        unsafe_allow_html=True
    )

    st.write(
        """
        **Alzheimer’s disease** is a progressive brain disorder that affects memory,
        thinking, and behavior. Early detection can help with timely treatment
        and management.

        This app uses **Transfer Learning (ResNet50)** to classify brain MRI images into:
        - **✅ NonDemented (Healthy)**
        - **🟡 Very Mild Demented**
        - **🟠 Mild Demented**
        - **🔴 Moderate Demented**
        """
    )

    st.image(
        "Alzheimer.png",
        caption="Example MRI Scan Showing Brain Regions",
        use_container_width=True
    )

    st.info("👉 Go to the **Detection** page from the left sidebar to upload an MRI image and get predictions.")


elif page == "Detection":
    st.markdown(
        "<h1 style='text-align: center;'>🧠 Alzheimer’s MRI Classification</h1>",
        unsafe_allow_html=True
    )
    st.write(
        "Upload a brain MRI image below. The model will classify the image into one of four categories."
    )

    uploaded_file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Image", use_container_width=True)

        img_array = preprocess_image(image)
        prediction = model.predict(img_array)[0]
        predicted_idx = np.argmax(prediction)
        predicted_class = idx_to_class[predicted_idx]
        probability = prediction[predicted_idx] * 100

        message, color = status_messages[predicted_class]

        st.markdown(
            f"<h3 style='color:{color}; text-align:center;'>{message}</h3>",
            unsafe_allow_html=True
        )
        st.info(
            "For guidance on memory or cognitive health, please seek advice from a qualified healthcare professional."
        )