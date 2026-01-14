import streamlit as st
import joblib
import pandas as pd
import os

# --- Load Model (Dictionary isi 4 Model) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "all_models.pkl")

try:
    models_dict = joblib.load(model_path)
except FileNotFoundError:
    st.error("❌ File 'all_models.pkl' tidak ditemukan. Jalankan script training dulu!")
    st.stop()

st.title("🏡 Predict House Price")
st.write("Aplikasi prediksi harga rumah dengan multi-model comparison.")

# --- Sidebar: Pemilihan Model ---
st.sidebar.header("⚙️ Konfigurasi Model")
model_options = list(models_dict.keys())
selected_model_name = st.sidebar.selectbox(
    "Pilih Algoritma Model:", 
    model_options,
    index=2  # Default ke Random Forest (biasanya urutan ke-3)
)

# Ambil model yang dipilih user
current_model = models_dict[selected_model_name]

# --- Input User ---
inputs = {}

# Layout kolom agar lebih rapi
col1, col2 = st.columns(2)

with col1:
    inputs["GrLivArea"] = st.number_input("Luas Bangunan (sqft)", min_value=0, value=1500)
    inputs["LotArea"] = st.number_input("Luas Tanah (sqft)", min_value=0, value=5000)
    inputs["TotalBsmtSF"] = st.number_input("Luas Basement (sqft)", min_value=0, value=1000)
    inputs["BedroomAbvGr"] = st.number_input("Jumlah Kamar Tidur", min_value=0, value=3)
    inputs["FullBath"] = st.number_input("Jumlah Kamar Mandi", min_value=0, value=2)

with col2:
    inputs["TotRmsAbvGrd"] = st.number_input("Total Ruangan", min_value=0, value=6)
    inputs["OverallQual"] = st.slider("Kualitas Rumah (1-10)", 1, 10, 5)
    inputs["OverallCond"] = st.slider("Kondisi Rumah (1-10)", 1, 10, 5)
    inputs["KitchenQual"] = st.slider("Kualitas Dapur (1-5)", 1, 5, 3)
    inputs["GarageCars"] = st.number_input("Kapasitas Garasi (mobil)", min_value=0, value=2)
    inputs["GarageArea"] = st.number_input("Luas Garasi (sqft)", min_value=0, value=400)

# List Neighborhood (Hanya nama string, tidak perlu mapping angka lagi)
neighborhood_list = [
    "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr", "Crawfor", 
    "Edwards", "Gilbert", "IDOTRR", "MeadowV", "Mitchel", "NAmes", "NPkVill", 
    "NWAmes", "NoRidge", "NridgHt", "OldTown", "SWISU", "Sawyer", "SawyerW", 
    "Somerst", "StoneBr", "Timber", "Veenker"
]

selected_location = st.selectbox("Lokasi / Neighborhood", neighborhood_list)

# PENTING: Kita masukkan string asli (misal 'CollgCr'), 
# Model Pipeline akan otomatis mengubahnya jadi angka (Encoding).
inputs["Neighborhood"] = selected_location 

# --- Prediksi ---
if st.button(f"Hitung Harga dengan {selected_model_name}"):
    # Buat DataFrame dari input
    df_input = pd.DataFrame([inputs])
    
    try:
        # Prediksi langsung (Pipeline di dalam model akan mengurus preprocessing)
        prediction = current_model.predict(df_input)[0]
        
        st.divider()
        st.subheader("Hasil Prediksi")
        st.success(f"💰 Estimasi Harga: ${prediction:,.2f}")
        st.caption(f"Diprediksi menggunakan algoritma: **{selected_model_name}**")
        
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
        st.info("Tips: Pastikan nama kolom input sama persis dengan yang ada di training.")