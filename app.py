import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Cargar el modelo
model = load_model("mnist_ejercicio.keras")

st.title("Clasificador de dígitos MNIST")
st.write("Sube una imagen de un dígito manuscrito (0–9) en escala de grises.")

uploaded_file = st.file_uploader(
    "Sube una imagen (28x28)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((28, 28))

    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28)

    st.image(image, caption="Imagen cargada", use_column_width=True)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.write("Predicción:", predicted_class)

    st.write("Probabilidades:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{i}: {prob:.2%}")
