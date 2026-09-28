# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# from huggingface_hub import hf_hub_download


# HF_TOKEN = st.secrets["HF_TOKEN"]
# @st.cache_resource
# def load_model():
#     model_path = hf_hub_download(
#         repo_id="Eyaddddddd/AlzheimerClassifierTL",
#         filename="AlzheimerClassifierTL.h5",
#         token=HF_TOKEN
#     )
#     # return tf.keras.models.load_model(model_path)
#     return tf.keras.models.load_model(model_path, compile=False)

# model = load_model()


# class_mapping = {
#     "MildDemented": 0,
#     "ModerateDemented": 1,
#     "NonDemented": 2,
#     "VeryMildDemented": 3
# }
# idx_to_class = {v: k for k, v in class_mapping.items()}


# status_messages = {
#     "NonDemented":  ("✅ Likely Healthy / No Dementia detected", "green"),
#     "VeryMildDemented": ("🟡 Signs of Very Mild Dementia", "orange"),
#     "MildDemented": ("🟠 Signs of Mild Dementia", "darkorange"),
#     "ModerateDemented": ("🔴 Signs of Moderate Dementia", "darkred")
# }


# def preprocess_image(image):
#     IMG_SIZE = (224, 224)
#     image = image.convert("RGB")
#     image = image.resize(IMG_SIZE)
#     img_array = np.array(image)
#     img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array


# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Go to", ["Home", "Detection"])


# if page == "Home":
#     st.markdown(
#         "<h1 style='text-align: center;'>🧠 Alzheimer’s Detection App</h1>",
#         unsafe_allow_html=True
#     )
#     st.markdown(
#         "<h3 style='text-align: center;'>Transfer Learning with ResNet50</h3>",
#         unsafe_allow_html=True
#     )

#     st.write(
#         """
#         **Alzheimer’s disease** is a progressive brain disorder that affects memory,
#         thinking, and behavior. Early detection can help with timely treatment
#         and management.

#         This app uses **Transfer Learning (ResNet50)** to classify brain MRI images into:
#         - **✅ NonDemented (Healthy)**
#         - **🟡 Very Mild Demented**
#         - **🟠 Mild Demented**
#         - **🔴 Moderate Demented**
#         """
#     )

#     st.image(
#         "Alzheimer.png",
#         caption="Example MRI Scan Showing Brain Regions",
#         use_container_width=True
#     )

#     st.info("👉 Go to the **Detection** page from the left sidebar to upload an MRI image and get predictions.")


# elif page == "Detection":
#     st.markdown(
#         "<h1 style='text-align: center;'>🧠 Alzheimer’s MRI Classification</h1>",
#         unsafe_allow_html=True
#     )
#     st.write(
#         "Upload a brain MRI image below. The model will classify the image into one of four categories."
#     )

#     uploaded_file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded MRI Image", use_container_width=True)

#         img_array = preprocess_image(image)
#         prediction = model.predict(img_array)[0]
#         predicted_idx = np.argmax(prediction)
#         predicted_class = idx_to_class[predicted_idx]
#         probability = prediction[predicted_idx] * 100

#         message, color = status_messages[predicted_class]

#         st.markdown(
#             f"<h3 style='color:{color}; text-align:center;'>{message}</h3>",
#             unsafe_allow_html=True
#         )
#         st.info(
#             "For guidance on memory or cognitive health, please seek advice from a qualified healthcare professional."
#         )




import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download


HF_TOKEN = st.secrets["HF_TOKEN"]


def build_model():
    # Same architecture as training; weights=None because we load our own trained weights
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights=None,
        input_shape=(224, 224, 3),
        pooling="max",   # must match training (notebook cell 38)
    )
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(224, 224, 3)),
        base,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.45),
        tf.keras.layers.Dense(4, activation="softmax"),
    ])
    return model


@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Eyaddddddd/AlzheimerClassifierTL",
        filename="AlzheimerClassifierTL.h5",
        token=HF_TOKEN
    )
    model = build_model()
    model.load_weights(model_path)
    return model


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
    # flow_from_dataframe resizes with "nearest" by default, so match it here
    image = image.resize(IMG_SIZE, Image.NEAREST)
    img_array = np.array(image, dtype=np.float32)
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
        width="stretch"
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
        st.image(image, caption="Uploaded MRI Image", width="stretch")

        img_array = preprocess_image(image)
        prediction = model.predict(img_array, verbose=0)[0]
        predicted_idx = int(np.argmax(prediction))
        predicted_class = idx_to_class[predicted_idx]
        probability = prediction[predicted_idx] * 100

        message, color = status_messages[predicted_class]

        st.markdown(
            f"<h3 style='color:{color}; text-align:center;'>{message}</h3>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='text-align:center;'>Confidence: <b>{probability:.2f}%</b></p>",
            unsafe_allow_html=True
        )
        st.info(
            "For guidance on memory or cognitive health, please seek advice from a qualified healthcare professional."
        )