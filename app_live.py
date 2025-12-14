import streamlit as st
import joblib
import time
import pandas as pd
from datetime import datetime
# Anda perlu menginstal pustaka ini: pip install google-api-python-client
from googleapiclient.discovery import build 

# --- 1. Konfigurasi dan Pemuatan Model ---

# PENTING: Pastikan jalur file sudah benar
MODEL_PATH = "models/model_svm (2).pkl"
TFIDF_PATH = "models/tfidf.pkl"

@st.cache_resource
def load_resources():
    try:
        model = joblib.load(MODEL_PATH)
        tfidf = joblib.load(TFIDF_PATH)
        return model, tfidf
    except FileNotFoundError:
        st.error(f"Gagal memuat model atau vectorizer dari {MODEL_PATH} atau {TFIDF_PATH}. Pastikan file ada.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat sumber daya: {e}")
        st.stop()

model, tfidf = load_resources()
label_map = {0: "Ham", 1: "Spam"}

# Data dummy untuk simulasi live chat
SIMULATED_COMMENTS = [
    "Videonya sangat informatif, terima kasih banyak!",  # Ham
    "KLIK DI SINI DAPATKAN VOUCHER GRATIS LANGSUNG!!",  # Spam
    "Terbaik lah pokoknya, subscribe!", # Ham
    "Beli followers instan, kunjungi link kami di bio.", # Spam
    "Ada yang tahu ini bahas apa ya?", # Ham
    "KEREN! Keren banget! 👍", # Ham
    "Dapatkan penghasilan jutaan tanpa modal, WA: 0812345678", # Spam
    "Setuju dengan komentator di atas.", # Ham
    " bisa main gaj tolol.",
]

# --- 2. Fungsi API YouTube (Perlu Kunci Asli) ---

# Global variable untuk menyimpan nextPageToken agar tidak mengambil chat yang sama berulang kali
# (Ini harus diinisialisasi dalam st.session_state jika di production)
if 'next_page_token' not in st.session_state:
    st.session_state.next_page_token = None

def fetch_youtube_comments(api_key, live_chat_id):
    """
    Fungsi untuk mengambil komentar live chat dari YouTube API.
    """
    # Mengambil token dari session state
    current_token = st.session_state.next_page_token 
    
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        # Panggil API liveChatMessages.list
        request = youtube.liveChatMessages().list(
            liveChatId=live_chat_id,
            part='snippet,authorDetails',
            # Gunakan token dari session state
            pageToken=current_token 
        )
        response = request.execute()
        
        # Simpan token baru ke session state
        st.session_state.next_page_token = response.get('nextPageToken')

        # ... (Sisa fungsi untuk ekstraksi komentar)
        comments = []
        for item in response.get('items', []):
            text = item['snippet']['displayMessage']
            comments.append(text)
            
        return comments

    except Exception as e:
        # Menambahkan pengecualian untuk error invalid token agar bisa di-debug
        if "page token is not valid" in str(e):
            st.error("TOKEN INVALID: Mencoba memulai ulang stream. Harap tekan Mulai kembali.")
            stop_stream()
        st.warning(f"Gagal mengambil data dari YouTube API: {e}. Pastikan API Key dan Live Chat ID benar.")
        return []

# --- 3. Fungsi Klasifikasi ---

def classify_comment(text):
    """Mengubah teks menjadi vektor dan memprediksi menggunakan model."""
    if not text.strip():
        return "N/A"
    
    vector = tfidf.transform([text])
    prediction = model.predict(vector)[0]
    
    return label_map.get(prediction, "Unknown")

# --- 4. Fungsi Styling Pandas (Perbaikan Warna/Tebal) ---

def highlight_prediction(val):
    """Fungsi untuk mewarnai sel berdasarkan nilai 'Prediksi'."""
    if val == 'Spam':
        # Warna latar belakang merah, teks tebal
        return 'background-color: #ffcccc; font-weight: bold; color: black'
    elif val == 'Ham':
        # Warna latar belakang hijau, teks tebal
        return 'background-color: #ccffcc; font-weight: bold; color: black'
    return ''

# --- 5. Logika Tombol dan Session State ---

# Inisialisasi state jika belum ada
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'comment_log' not in st.session_state:
    st.session_state.comment_log = pd.DataFrame(columns=['Waktu', 'Komentar', 'Prediksi', 'Warna'])
if 'spam_count' not in st.session_state:
    st.session_state.spam_count = 0
if 'ham_count' not in st.session_state:
    st.session_state.ham_count = 0
if 'total_comments' not in st.session_state:
    st.session_state.total_comments = 0
if 'comment_index' not in st.session_state:
    st.session_state.comment_index = 0

def start_stream():
    st.session_state.is_running = True
    
def stop_stream():
    st.session_state.is_running = False

# --- 6. Tampilan Streamlit Utama ---

st.title("🔴 Live Dashboard: Klasifikasi Komentar YouTube")
st.write("Analisis Komentar Real-time menggunakan Model SVM.")

# Input API dan Live Chat ID
st.sidebar.header("Konfigurasi YouTube API")
api_key = st.sidebar.text_input("YouTube API Key:", type="password")
live_chat_id = st.sidebar.text_input("Live Chat ID:")

# Opsi Data (Simulasi vs Nyata)
data_source = st.sidebar.radio(
    "Sumber Data:",
    ("Simulasi (Offline)", "YouTube API (Online)"),
    index=0
)

# Tombol Start/Stop
col_start, col_stop, col_status = st.columns([1, 1, 3])

col_start.button("▶️ Mulai Stream", on_click=start_stream, type="primary", disabled=st.session_state.is_running)
col_stop.button("⏹️ Hentikan Stream", on_click=stop_stream, disabled=not st.session_state.is_running)

if st.session_state.is_running:
    col_status.success(f"Stream Aktif ({data_source})")
else:
    col_status.warning("Stream Tidak Aktif. Tekan Mulai.")
    
st.header("Metrik Live")
metrics_placeholder = st.empty()
st.header("Log Komentar Terbaru")
live_log_placeholder = st.empty()


# --- 7. Loop Esekusi Live ---

if st.session_state.is_running:
    
    # Pilih sumber data
    if data_source == "YouTube API (Online)":
        if not api_key or not live_chat_id:
            st.error("Masukkan API Key dan Live Chat ID untuk memulai stream YouTube.")
            stop_stream()
            st.rerun() # Rerun untuk update status
            
        # Ambil data dari YouTube (Menggunakan waktu refresh yang ditetapkan YouTube)
        new_comments = fetch_youtube_comments(api_key, live_chat_id)
        # YouTube merekomendasikan refresh sekitar 5-10 detik
        sleep_time = 5 
        
    else: # Data Simulasi
        # Ambil komentar dari array simulasi
        new_comments = [SIMULATED_COMMENTS[st.session_state.comment_index]]
        st.session_state.comment_index = (st.session_state.comment_index + 1) % len(SIMULATED_COMMENTS)
        sleep_time = 2 # Lebih cepat untuk simulasi

    
    if new_comments:
        # Proses semua komentar baru
        for new_comment_text in new_comments:
            
            prediction = classify_comment(new_comment_text)
            
            if prediction == "Spam":
                st.session_state.spam_count += 1
            elif prediction == "Ham":
                st.session_state.ham_count += 1
                
            st.session_state.total_comments += 1
            
            # Buat baris data baru
            new_data = pd.DataFrame({
                'Waktu': [datetime.now().strftime("%H:%M:%S")],
                'Komentar': [new_comment_text],
                'Prediksi': [prediction],
            })
            
            # Tambahkan data baru ke log dan batasi 10 entri terbaru
            st.session_state.comment_log = pd.concat(
                [new_data, st.session_state.comment_log]
            ).head(10).reset_index(drop=True)

    # --- Perbarui Tampilan Metrik ---
    with metrics_placeholder.container():
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Komentar", st.session_state.total_comments)
        
        # Hindari pembagian dengan nol
        spam_percent = round((st.session_state.spam_count / st.session_state.total_comments) * 100) if st.session_state.total_comments > 0 else 0
        ham_percent = round((st.session_state.ham_count / st.session_state.total_comments) * 100) if st.session_state.total_comments > 0 else 0
        
        col2.metric("Spam Terdeteksi", st.session_state.spam_count, delta=f"{spam_percent}%", delta_color="inverse")
        col3.metric("Ham", st.session_state.ham_count, delta=f"{ham_percent}%", delta_color="normal")
        
    # --- Perbarui Tampilan Tabel Log (dengan Styling) ---
    
    # Gunakan .style untuk mewarnai dan menebalkan teks
    display_df = st.session_state.comment_log.copy()
    
    live_log_placeholder.dataframe(
        display_df.style.applymap(
            highlight_prediction, 
            subset=['Prediksi']
        ),
        hide_index=True,
        use_container_width=True
    )
    
    # Jeda loop dan minta Streamlit untuk menjalankan ulang
    time.sleep(sleep_time)
    st.rerun()