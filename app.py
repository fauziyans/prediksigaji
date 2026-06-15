import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. SETTING HALAMAN UTAMA
st.set_page_config(page_title="Software Engineer Salary Predictor", layout="centered")
st.title("💼 Software Engineer Salary Predictor")
st.write("Aplikasi prediksi gaji tahunan berbasis Machine Learning (Regresi Linear).")
st.markdown("---")

# 2. LOAD MODEL DAN ENCODER YANG SUDAH DISIMPAN
@st.cache_resource # Agar model tidak di-load berulang kali setiap klik tombol
def load_components():
    with open('model_regresi.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('kolom_training.pkl', 'rb') as f:
        kolom_training = pickle.load(f)
    return model, encoder, kolom_training

try:
    model, encoder, kolom_training = load_components()
except FileNotFoundError:
    st.error("File model atau encoder tidak ditemukan. Pastikan Anda sudah menjalankan sel export di Jupyter Notebook!")
    st.stop()

# 3. MEMBUAT FORM INPUT USER DI UI
st.subheader("Masukkan Profil Kandidat:")

col1, col2 = st.columns(2)

with col1:
    experience = st.number_input("Pengalaman Kerja (Tahun):", min_value=0, max_value=50, value=2, step=1)
    
    # Ambil opsi kategori langsung dari encoder yang sudah dilatih sebelumnya
    education_options = encoder.categories_[0].tolist()
    education = st.selectbox("Tingkat Pendidikan:", education_options)

with col2:
    # Cari list negara dari kolom_training (mencari teks setelah 'country_')
    # Jika tidak ada, sediakan opsi default (sesuaikan dengan data Anda)
    country_options = ['Germany', 'USA', 'India', 'Canada', 'United Kingdom'] 
    country = st.selectbox("Negara Domisili:", country_options)
    
    company_size_options = encoder.categories_[1].tolist()
    company_size = st.selectbox("Ukuran Perusahaan:", company_size_options)

# 4. TOMBOL PROSES PREDIKSI
st.markdown("<br>", unsafe_allow_html=True)
if st.button("Hitung Estimasi Gaji", type="primary"):
    
    # A. Bungkus input data user ke dalam DataFrame satu baris
    input_data = pd.DataFrame([{
        'experience': experience,
        'education': education,
        'company_size': company_size
    }])
    
    # B. Jalankan Ordinal Encoding untuk pendidikan dan skala perusahaan
    input_data[['education', 'company_size']] = encoder.transform(input_data[['education', 'company_size']])
    
    # C. SOLUSI UTAMA: Membuat struktur kolom training dasar yang bernilai 0 semua
    input_ready = pd.DataFrame(0, index=[0], columns=kolom_training)
    
    # D. Masukkan nilai pengalaman, pendidikan, dan ukuran perusahaan yang sudah di-encode
    input_ready['experience'] = experience
    input_ready['education'] = input_data['education'].iloc[0]
    input_ready['company_size'] = input_data['company_size'].iloc[0]
    
    # E. Tembak angka 1 khusus untuk kolom negara yang dipilih user
    nama_kolom_negara = f"country_{country}"
    if nama_kolom_negara in input_ready.columns:
        input_ready[nama_kolom_negara] = 1
    # Jika tidak ada di columns, berarti negara tersebut adalah 'drop_first' (baseline), 
    # biarkan kolom negara lain bernilai 0 (sudah benar secara teori regresi)

    # F. Prediksi nilai target menggunakan model Regresi Linear
    hasil_prediksi = model.predict(input_ready)[0]
    
    # G. Tampilkan hasil prediksi ke layar UI Streamlit
    st.markdown("---")
    st.subheader("🎯 Hasil Analisis Model:")
    
    if hasil_prediksi < 0:
        hasil_prediksi = 0
        
    st.success(f"Estimasi Nilai Gaji Rekomendasi: *${hasil_prediksi:,.2f} USD / Tahun*")
    st.info("💡 Catatan: Model memiliki batas rentang deviasi rata-rata (RMSE) sekitar ±$17,046.")