import os
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

MODEL_PATH = '../data/alexnet.h5'
IMAGE_FOLDER = '../data/'

# ---------------------
# FUNCTIONS
# ---------------------
def load_selected_model(selection):
    if selection == "AlexNet":
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    elif selection == "VGG16":
        model = tf.keras.applications.VGG16(weights="imagenet", include_top=False)
    elif selection == "VGG19":
        model = tf.keras.applications.VGG19(weights="imagenet", include_top=False)
    elif selection == "ResNet101":
        model = tf.keras.applications.ResNet101(weights="imagenet", include_top=False)
    elif selection == "ResNet152":
        model = tf.keras.applications.ResNet152(weights="imagenet", include_top=False)
    elif selection == "ResNet50":
        model = tf.keras.applications.ResNet50(weights="imagenet", include_top=False)
    else:
        raise ValueError("Unknown model selected.")
    return model

def list_conv_layers(model):
    return [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]

def show_filters(model):
    import matplotlib.colors as mcolors

    first_conv = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            first_conv = layer
            break
    if first_conv is None:
        st.warning("No Conv2D layer found.")
        return

    filters, _ = first_conv.get_weights()
    n_filters = filters.shape[3]
    n_rows = min(8, n_filters)
    n_cols = int(np.ceil(n_filters / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*1.5, n_rows*1.5))

    for i in range(n_filters):
        f = filters[:, :, :, i]

        # Normalize each filter to 0-1 per channel
        f_min = f.min()
        f_max = f.max()
        f = (f - f_min) / (f_max - f_min + 1e-6)

        # Gamma correction to enhance contrast
        f = np.power(f, 0.7)

        # Convert grayscale to RGB if needed
        if f.shape[2] == 1:
            f = np.repeat(f, 3, axis=2)
        elif f.shape[2] != 3:
            f = np.concatenate([f] * 3, axis=2)

        ax = axes.flat[i]
        ax.imshow(f)
        ax.axis('off')

    for j in range(i+1, n_rows * n_cols):
        axes.flat[j].axis('off')

    plt.tight_layout()
    st.pyplot(fig)


def preprocess_image(img, model, model_name):
    input_shape = (224, 224)
        # input_shape = model.input_shape[1:3]
    img = img.resize(input_shape)
    img = np.array(img).astype(np.float32) / 255.0
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    img = np.expand_dims(img, axis=0)
    return img


def show_feature_maps(model, img_array, layer_name):
    intermediate_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    feature_maps = intermediate_model.predict(img_array)[0]
    n_maps = feature_maps.shape[-1]
    n_rows = min(8, n_maps)
    n_cols = int(np.ceil(n_maps / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*1.5, n_rows*1.5))
    for i in range(n_maps):
        fmap = feature_maps[:, :, i]
        fmap -= fmap.min()
        fmap /= fmap.max() + 1e-5
        ax = axes.flat[i]
        ax.imshow(fmap, cmap='Greys_r')
        ax.axis('off')
    for j in range(i+1, n_rows * n_cols):
        axes.flat[j].axis('off')
    st.pyplot(fig)

# ---------------------
# STREAMLIT APP
# ---------------------
st.title("🔎 Interactive CNN Analysis")

model_choice = st.selectbox("Select model", ["AlexNet", "VGG16", "VGG19", "ResNet50", "ResNet101", "ResNet152"])
model = load_selected_model(model_choice)
st.success(f"{model_choice} loaded.")
st.info(f"Model input size: {model.input_shape[1:]}")

st.header("1️⃣ Visualize first Conv filters")
if st.button("Show filters of first Conv2D layer"):
    show_filters(model)

st.header("2️⃣ Upload or select image to analyze")
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
selected_file = st.selectbox("Select image from data folder", image_files)
uploaded_file = st.file_uploader("...or upload an image", type=['png', 'jpg', 'jpeg'])

img = None
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
elif selected_file:
    img_path = os.path.join(IMAGE_FOLDER, selected_file)
    img = Image.open(img_path)
    st.image(img, caption=f"Selected Image: {selected_file}", use_column_width=True)

if img is not None:
    img_array = preprocess_image(img, model, model_choice)

    st.header("3️⃣ Feature Maps")
    conv_layers = list_conv_layers(model)
    if conv_layers:
        layer_choice = st.selectbox("Select Conv2D layer to visualize activations", conv_layers)
        if st.button("Show feature maps"):
            show_feature_maps(model, img_array, layer_choice)
    else:
        st.warning("No Conv2D layers found in model.")
