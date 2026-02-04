import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Configuración de la página
st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")

st.title("🖌️ Reconocimiento de Dígitos MNIST")
st.write("Sube una imagen de un número. IMPORTANTE: El modelo está entrenado con números BLANCOS sobre fondo NEGRO.")

# Cargar el modelo
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('modelo_mnist.keras')

try:
    model = load_my_model()
    st.success("✅ Modelo cargado y listo")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")

# Subida de imagen
uploaded_file = st.file_uploader("Sube tu imagen", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("📷 **Imagen Original**")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

    # --- PREPROCESAMIENTO ---
    # 1. Convertir a escala de grises
    img_gray = image.convert('L')
    
    # 2. Redimensionar a 28x28 (Tamaño exacto de MNIST)
    img_resized = img_gray.resize((28, 28))
    
    # 3. Convertir a array
    img_array = np.array(img_resized)
    
    # 4. OPCIÓN CRÍTICA: Invertir colores
    # Añadimos un checkbox para que tú controles esto
    st.write("---")
    st.write("⚙️ **Ajustes de Imagen**")
    invertir = st.checkbox("Invertir colores (Úsalo si tu imagen es Tinta Negra sobre Fondo Blanco)", value=True)
    
    if invertir:
        img_array = 255 - img_array

    # 5. Normalizar
    img_processed = img_array.reshape(1, 784).astype('float32') / 255

    with col2:
        st.write("👁️ **Lo que ve el modelo**")
        # Mostramos la imagen tal cual entra a la red (importante para debug)
        st.image(img_array, clamp=True, width=150, caption="Si esto no parece un número blanco sobre negro, invierte los colores.")

    # --- PREDICCIÓN ---
    if st.button('🔮 Predecir Número'):
        prediction = model.predict(img_processed)
        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        st.divider()
        st.markdown(f"<h1 style='text-align: center; color: green;'>Es un: {digit}</h1>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>Confianza: {confidence:.2f}%</h3>", unsafe_allow_html=True)
        
        # Gráfica de barras
        st.write("Probabilidades por número:")
        st.bar_chart(prediction[0])
