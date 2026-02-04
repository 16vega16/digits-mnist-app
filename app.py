import streamlit as st
import tensorflow as keras
import numpy as np
from PIL import Image

st.title("Reconocimiento de Dígitos MNIST")
st.write("Sube una imagen de un número manuscrito (0-9) y el modelo intentará adivinar cuál es.")

@st.cache_resource
def load_my_model():
    return keras.models.load_model('modelo_mnist.h5')

try:
    model = load_my_model()
    st.success("¡Modelo cargado correctamente!")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")

uploaded_file = st.file_uploader("Sube tu imagen (png, jpg, jpeg)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen subida', use_column_width=False, width=150)
    
    img_gray = image.convert('L')
    img_resized = img_gray.resize((28, 28))
    img_array = np.array(img_resized)
    img_array = 255 - img_array
    
    img_processed = img_array.reshape(1, 784).astype('float32') / 255

    if st.button('Predecir'):
        prediction = model.predict(img_processed)
        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        st.markdown(f"### 🤖 El modelo predice que es un: **{digit}**")
        st.write(f"Confianza: {confidence:.2f}%")
        
        # Opcional: Mostrar gráfico de barras con las probabilidades
        st.bar_chart(prediction[0])
