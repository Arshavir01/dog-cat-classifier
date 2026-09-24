from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow_hub as hub
import tf_keras
from PIL import Image, ImageOps

BASE = Path(__file__).parent

st.set_page_config(page_title="Dog vs Cat Classifier", page_icon="🐶")


@st.cache_resource  # load the model only once, not on every click
def load_model():
    return tf_keras.models.load_model(
        str(BASE / "dog_cat_model.h5"),
        custom_objects={"KerasLayer": hub.KerasLayer},
    )


def predict(img):
    img = ImageOps.exif_transpose(img)            # fix rotated phone photos
    img = img.convert("RGB").resize((224, 224))   # same size as training
    arr = np.asarray(img, dtype=np.float32)
    arr = arr[:, :, ::-1]                         # RGB -> BGR (training used cv2.imread)
    arr = arr / 255.0                             # same scaling as training
    logits = model.predict(arr[np.newaxis, ...], verbose=0)[0]
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()                   # softmax -> probabilities
    label = int(np.argmax(probs))
    return ("Cat 🐱" if label == 0 else "Dog 🐶"), float(probs[label])


st.title("Dog vs Cat Classifier")
st.write("Upload a photo and a MobileNetV2 transfer learning model will tell you "
         "whether it shows a dog or a cat.")
st.caption("Trained on 2,000 images from the Kaggle Dogs vs Cats dataset. "
           "Test accuracy: 98%. Photos of other animals will still be labelled dog or cat.")

with st.spinner("Loading model..."):
    model = load_model()

uploaded = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    try:
        image = Image.open(uploaded)
        image.thumbnail((1024, 1024))             # shrink large phone photos early
        st.image(ImageOps.exif_transpose(image), use_container_width=True)

        label, confidence = predict(image)
        st.success(f"This is a **{label}** ({confidence:.0%} confidence)")
    except Exception:
        st.error("Could not read this image. Please try a JPG or PNG photo.")
