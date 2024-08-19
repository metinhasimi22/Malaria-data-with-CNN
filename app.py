import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Hastalık sınıfları
sınıflar = ['Hastalıklı', 'Sağlıklı']

# Modeli yükleme
model = load_model('model95.h5')  # Modelinizin yolunu buraya ekleyin
model.summary()

def process_img(img):
    img = img.resize((128, 128), Image.LANCZOS)  # 128x128 piksel boyutuna dönüştürme
    img = np.array(img) / 255.0  # Normalize etme
    img = np.expand_dims(img, axis=0)  # Resme boyut ekleme
    return img

st.title("Malaria Hastalığı Sınıflandırması :date:")
st.write(
    "Bir mikroskop resmi seçin ve modelimiz, bu resmin **Malaria** hastalığı gösterip göstermediğini tahmin etsin. 🖼️📊\n"
    "Upload an image and the model will predict whether the image shows **Malaria** or not."
)

# Stil ayarları
st.markdown("""
<style>
    .reportview-container {
        background: #F0F2F6;
    }
    .sidebar .sidebar-content {
        background: #E0E0E0;
    }
    .css-18e3th9 {
        font-size: 1.25em;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

file = st.file_uploader("Resim Yükle & Bir resim seçiniz", type=['png', 'jpg', 'jpeg'])

if file is not None:
    img = Image.open(file)
    st.image(img, caption="Yüklenen Resim", use_column_width=True)
    
    result = process_img(img)
    prediction = model.predict(result)
    prediction_class = np.argmax(prediction)  # En yüksek tahmin edilen sınıf

    # Sınıf isimleri
    result_text = sınıflar[prediction_class]

    st.write(f"**Sonuç:** {result_text}")
else:
    st.write("Lütfen bir resim yükleyin.")
