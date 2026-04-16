import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("model.h5")

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

st.title("🖼️ CIFAR-10 Image Classifier")

file = st.file_uploader("Upload Image", type=["jpg","png"])

if file:
    img = Image.open(file).resize((32,32))
    st.image(img)

    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_index = np.argmax(pred)

    st.write("Prediction:", class_names[class_index])
    st.write("Confidence:", float(np.max(pred))*100, "%")