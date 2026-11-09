# SmartClinic No-Show Predictor

Aplikasi Streamlit untuk memperkirakan kemungkinan pasien rawat jalan **hadir** atau **tidak hadir** pada jadwal
konsultasi, menggunakan model PyTorch (MLP) yang disimpan dalam file `model/model_no_show.pth`.
Proyek ini adalah **Tugas Besar Machine Learning - Part 3** (lanjutan Part 1 dan Part 2).

> **Catatan akademik:** hasil prediksi berupa estimasi probabilitas, bukan kepastian, dan tidak boleh menjadi satu-satunya
> dasar keputusan klinis atau pembatalan janji temu otomatis.

## Tujuan proyek
- Part 1: merancang ide SmartClinic No-Show Predictor (kerangka CRISP-DM).
- Part 2: preprocessing, pemodelan dan tuning Logistic Regression, Decision Tree, Random Forest (scikit-learn).
- Part 3: portasi ke **PyTorch**, penyimpanan model `.pth`, dan aplikasi dengan User Interface.

## Perbedaan Part 2 dan Part 3
Part 2 memakai model scikit-learn. Part 3 menambahkan model **MLP PyTorch** pada dataset, pembersihan data, pembagian
data (80/20, stratified, `random_state=42`) dan preprocessing yang sama, agar model dapat disimpan sebagai `.pth` dan
diintegrasikan ke aplikasi. Model PyTorch adalah model baru, sehingga metrik dievaluasi ulang
(lihat `reports/comparison_part2_vs_part3.csv`).

## Teknologi
Python 3.10+, PyTorch, Streamlit, pandas, NumPy, pytest. Untuk pelatihan ulang: scikit-learn dan matplotlib.

## Struktur folder
```text
smartclinic-no-show-predictor/
├── app.py                     # aplikasi Streamlit
├── model/model_no_show.pth    # checkpoint PyTorch (bobot + konfigurasi + preprocessing)
├── src/
│   ├── preprocessing.py       # pembersihan data, fitur waktu, encoding/scaling
│   ├── model_architecture.py  # arsitektur MLP
│   └── prediction.py          # memuat .pth, validasi input, prediksi
├── notebooks/Tugas_Besar_Machine_Learning_Part3.ipynb   # notebook Google Colab (training + simpan .pth + download)
├── scripts/                   # train.py, verify_model.py, make_report.py
├── reports/                   # comparison_part2_vs_part3.csv, evaluation/ (metrik dan grafik)
├── tests/                     # pengujian otomatis (pytest)
├── data/                      # letakkan CSV Kaggle di sini (opsional, lihat data/README.md)
├── assets/  docs/screenshots/
├── requirements.txt  requirements-train.txt  .gitignore  README.md
└── .streamlit/config.toml
```

## Persyaratan sistem
Python 3.10 atau lebih baru (diuji pada 3.12), pip, RAM minimal 2 GB. GPU tidak diperlukan.

## Instalasi dan menjalankan di Windows (PowerShell)
```powershell
cd smartclinic-no-show-predictor
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```
Jika PowerShell menolak mengaktifkan venv: `Set-ExecutionPolicy -Scope Process Bypass`, atau gunakan Command Prompt
(`.venv\Scripts\activate.bat`). Peramban akan terbuka di `http://localhost:8501`.
Instalasi PyTorch versi CPU yang lebih ringan: lihat panduan di pytorch.org lalu jalankan `pip install -r requirements.txt` kembali.

## Instalasi dan menjalankan di Linux / macOS
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Cara menggunakan aplikasi
1. Isi data pasien: jenis kelamin, umur (0-100), kelurahan, kondisi (beasiswa, hipertensi, diabetes, alkoholisme, handicap 0-4) dan SMS pengingat.
2. Isi jadwal: tanggal dan waktu pendaftaran jadwal serta tanggal janji temu. `waiting_days`, `appt_weekday` dan `sched_hour` dihitung otomatis seperti pada Part 2.
3. Tekan **Prediksi** untuk melihat estimasi probabilitas hadir dan tidak hadir. Tombol **Hapus / ulangi input** mengembalikan nilai awal.
4. Aplikasi menolak input tidak valid (mis. janji temu sebelum tanggal pendaftaran) dan memberi peringatan bila input di luar rentang data latih (jeda > 179 hari, janji hari Sabtu/Minggu).

## Tentang file model `.pth`
Berisi `model_state_dict`, konfigurasi arsitektur (input 92, hidden 64 dan 32, dropout 0,3), parameter preprocessing
(rata-rata/skala, daftar kategori), urutan fitur, pemetaan kelas (0 = Hadir, 1 = Tidak hadir), ambang keputusan 0,5 dan
metrik uji. Aplikasi memuatnya dengan `torch.load(..., weights_only=True)`, `model.eval()` dan `torch.inference_mode()`.

**Mengganti dengan model hasil Google Colab:** jalankan notebook di folder `notebooks/` pada Colab, unduh
`model_no_show.pth`, lalu timpa `model/model_no_show.pth`. Tidak ada perubahan kode yang diperlukan.
Lokasi lain dapat diatur dengan variabel lingkungan `NOSHOW_MODEL_PATH`.

## Pengujian
```bash
python -m pytest tests -q
```
Uji metrik penuh (`test_metrik_uji_direproduksi_dari_pth`) hanya berjalan jika `data/KaggleV2-May-2016.csv` dan scikit-learn tersedia.

## Melatih ulang (opsional)
```bash
pip install -r requirements-train.txt
python scripts/train.py --data data/KaggleV2-May-2016.csv
```
Dataset: Kaggle, *Medical Appointment No Shows* (`KaggleV2-May-2016.csv`).

## Informasi pengembang
Yudi Ahmad Taqiyyudin - NPM 202343501555 - Kelas X7J - Teknik Informatika, Fakultas Teknik dan Ilmu Komputer,
Universitas Indraprasta PGRI (UNINDRA). Dosen: Ahmadi, S.Kom., M.T.I. Mata kuliah: Machine Learning.
