import streamlit as st 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model yang telah disimpan
model = load_model('model_katarak.h5')

# Daftar nama kelas
class_names = ['Normal', 'Katarak']  # Sesuaikan dengan dataset

# Judul aplikasi
st.title("Deteksi Katarak pada Retina Mata Manusia")
st.write("Upload gambar retina untuk mendeteksi apakah mengalami katarak atau tidak.")

# Upload gambar
uploaded_file = st.file_uploader("Upload gambar retina", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar Retina", use_column_width=True)

    # Preprocessing
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    prediction = model.predict(img_array)[0][0]
    predicted_class = 1 if prediction >= 0.5 else 0
    confidence = float(prediction) if predicted_class == 1 else 1 - float(prediction)

    # Tampilkan hasil
    st.markdown(f"### Prediksi: **{class_names[predicted_class]}**")
    st.markdown(f"### Kepercayaan: **{confidence * 100:.2f}%**")
