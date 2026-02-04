import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(page_title="MNIST Pro", layout="centered")

st.title("🖌️ Reconocimiento de Dígitos MNIST (Versión Pro)")
st.write("Sube tu imagen. Esta versión centra y ajusta el número automáticamente.")

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('modelo_mnist.keras')

try:
    model = load_my_model()
    st.success("✅ Modelo cargado")
except Exception as e:
    st.error(f"Error: {e}")

# Función mágica para procesar la imagen como le gusta a MNIST
def procesar_imagen_experta(image, invertir_colores=True, umbral=100):
    # 1. Escala de grises
    img = image.convert('L')
    
    # 2. Invertir si es necesario (Tinta negra -> Fondo negro)
    if invertir_colores:
        img = ImageOps.invert(img)
    
    # 3. Aplicar umbral (limpiar ruido): Todo lo que no sea muy blanco, se vuelve negro
    # Esto elimina sombras del papel o borrones
    img = img.point(lambda p: 255 if p > umbral else 0)
    
    # 4. Recortar al contenido (Bounding Box)
    # Buscamos dónde está el número y quitamos el sobrante negro
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
        
    # 5. Hacerla cuadrada (Padding) para no deformar
    w, h = img.size
    max_side = max(w, h)
    
    # Creamos un lienzo negro cuadrado
    new_img = Image.new('L', (max_side, max_side), 0) # 0 es negro
    
    # Pegamos el número en el centro
    offset_x = (max_side - w) // 2
    offset_y = (max_side - h) // 2
    new_img.paste(img, (offset_x, offset_y))
    
    # 6. Redimensionar a 20x20 (El estándar MNIST es número de 20px)
    new_img = new_img.resize((20, 20), Image.Resampling.LANCZOS)
    
    # 7. Pegar en lienzo de 28x28 (para tener el margen de 4px)
    final_img = Image.new('L', (28, 28), 0)
    final_img.paste(new_img, (4, 4))
    
    return final_img

uploaded_file = st.file_uploader("Sube imagen", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file)
    
    with col1:
        st.image(image, caption="Original", width=150)
        invertir = st.checkbox("Invertir colores (Negro sobre blanco)", value=True)
        # Slider para ajustar la sensibilidad del blanco/negro
        umbral = st.slider("Umbral de limpieza (Ajusta si se ve ruido)", 0, 255, 120)

    # Procesamos
    img_final = procesar_imagen_experta(image, invertir, umbral)
    
    # Convertimos a numpy para predecir
    img_array = np.array(img_final)
    img_processed = img_array.reshape(1, 784).astype('float32') / 255

    with col2:
        st.image(img_final.resize((150, 150), resample=0), caption="Lo que ve la IA (28x28)", width=150)
        st.info("👆 El número debe verse blanco, centrado y con margen negro.")

    if st.button('🔮 Predecir'):
        prediction = model.predict(img_processed)
        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        st.markdown(f"# Es un: {digit}")
        st.caption(f"Confianza: {confidence:.2f}%")
        st.bar_chart(prediction[0])
        
