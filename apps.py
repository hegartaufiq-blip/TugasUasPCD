import cv2
import numpy as np
import streamlit as st

# 1. Konfigurasi Halaman Utama Web
st.set_page_config(
    page_title="Sistem Sortir Telur Ayam",
    page_icon="🥚",
    layout="wide"
)

# 2. Komponen Kontrol Interaktif di Sidebar (Bilah Samping)
with st.sidebar:
    st.header("🥚 Kontrol Parameter PCD")
    st.write("Sesuaikan kecocokan sensor dengan foto telur Anda secara real-time.")
    st.markdown("---")
    
    # Slider untuk menambal lubang cahaya (Morfologi Closing)
    ukuran_kernel = st.slider("1. Pembersih Kilatan Cahaya", min_value=3, max_value=41, value=15, step=2, 
                              help="Makin besar angka, makin kuat menambal lubang cahaya di tengah telur.")
    
    # Slider untuk mengatur ambang batas deteksi garis retak Canny Edge
    sensitivitas_canny = st.slider("2. Sensitivitas Deteksi Retak", min_value=10, max_value=150, value=40, step=5,
                                   help="Makin kecil angka, makin peka sistem mendeteksi retakan tipis.")
    
    # Slider untuk menentukan toleransi jumlah piksel cacat
    batas_piksel_cacat = st.slider("3. Toleransi Piksel Defect", min_value=100, max_value=2000, value=450, step=50,
                                    help="Jika piksel retak melebihi angka ini, telur dinyatakan rusak/defect.")
    st.markdown("---")
    st.caption("Proyek UAS PCD - S1 Informatika UBS PPNI Mojokerto")

# 3. Fungsi Pemrosesan Citra Digital Dinamis
def sortir_telur(image_bytes, k_size, canny_th, defect_th):
    # Membaca gambar input
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Pre-processing: Reduksi Noise (CPMK2)
    blurred = cv2.GaussianBlur(img, (9, 9), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    # Segmentasi Biner Otsu (CPMK4)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Operasi Morfologi Closing Dinamis berdasarkan input Slider 1
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
    
    # Deteksi Tepi Canny Dinamis berdasarkan input Slider 2
    edges = cv2.Canny(gray, canny_th, canny_th * 2)
    
    # Mencari Kontur Luas Bentuk Telur
    contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    status_ukuran = "Tidak Terdeteksi"
    status_kualitas = "Bersih / Normal"
    img_hasil = img_rgb.copy()
    total_defect_pixels = 0
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        luas_piksel = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        
        # Penentuan Kategori Ukuran berdasarkan Luas Area Piksel
        if luas_piksel > 120000:
            status_ukuran = f"Grade A (Besar / {int(luas_piksel)} px)"
        elif luas_piksel > 60000:
            status_ukuran = f"Grade B (Sedang / {int(luas_piksel)} px)"
        else:
            status_ukuran = f"Grade C (Kecil / {int(luas_piksel)} px)"
            
        # Membuat Masking Area Kulit Dalam Telur
        mask_telur = np.zeros_like(edges)
        cv2.drawContours(mask_telur, [c], -1, 255, -1)
        
        # Pengikisan tepi masker agar garis lingkaran terluar kulit tidak dikira retak
        kernel_ero = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask_telur = cv2.erode(mask_telur, kernel_ero)
        
        # Ekstraksi piksel retakan/kotoran menggunakan operasi AND bitwise
        kotoran_retak = cv2.bitwise_and(edges, mask_telur)
        total_defect_pixels = cv2.countNonZero(kotoran_retak)
        
        # Penentuan Keputusan Akhir Kualitas Berdasarkan Slider 3
        if total_defect_pixels > defect_th:
            status_kualitas = f"Defect / Retak / Kotor ({total_defect_pixels} noise px)"
            cv2.rectangle(img_hasil, (x, y), (x + w, y + h), (255, 0, 0), 4)  # Kotak Merah jika defect
        else:
            status_kualitas = f"Bersih / Normal ({total_defect_pixels} noise px)"
            cv2.rectangle(img_hasil, (x, y), (x + w, y + h), (0, 255, 0), 4)  # Kotak Hijau jika normal
            
    return img_rgb, thresh_clean, img_hasil, status_ukuran, status_kualitas

# 4. Tampilan Antarmuka Dashboard Web (Gaya Baru Tanpa Warning Kuning)
st.title("🥚 Sistem Sortir Telur Ayam Otomatis Berbasis PCD")
st.write("Aplikasi Prototipe untuk Memenuhi Tugas UAS Pengolahan Citra Digital - S1 Informatika UBS PPNI")

uploaded_file = st.file_uploader("Unggah Foto Telur Ayam Anda di Sini...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    
    # Memanggil fungsi sortir dengan menyertakan nilai dari slider
    img_ori, img_thresh, img_out, ukuran, kualitas = sortir_telur(
        bytes_data, ukuran_kernel, sensitivitas_canny, batas_piksel_cacat
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img_ori, caption="1. Citra Asli Telur", use_container_width=True)
    with col2:
        st.image(img_thresh, caption="2. Hasil Segmentasi (Biner)", use_container_width=True)
    with col3:
        st.image(img_out, caption="3. Visualisasi Hasil Sortir", use_container_width=True)
        
    st.markdown("---")
    st.subheader("📋 Hasil Analisis Sistem Penyortiran:")
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric(label="Kategori Ukuran (Volume)", value=ukuran)
    with c2:
        if "Defect" in kualitas:
            st.error(f"Kondisi Cangkang: {kualitas}")
        else:
            st.success(f"Kondisi Cangkang: {kualitas}")
  