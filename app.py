"""SmartClinic No-Show Predictor - aplikasi Streamlit (Tugas Besar Machine Learning Part 3).

Jalankan:  streamlit run app.py
"""
import os
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from src.prediction import (build_input_frame, load_checkpoint, neighbourhood_options, predict_one,
                            validate_inputs)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("NOSHOW_MODEL_PATH", ROOT / "model" / "model_no_show.pth"))
COMPARISON_CSV = ROOT / "reports" / "comparison_part2_vs_part3.csv"
DISCLAIMER = ("Aplikasi ini merupakan proyek akademik. Hasil berupa estimasi probabilitas, bukan kepastian, "
              "dan tidak boleh menjadi satu-satunya dasar keputusan klinis atau pembatalan janji temu otomatis.")

st.set_page_config(page_title="SmartClinic No-Show Predictor", page_icon="🩺", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1150px; padding-top: 2rem;}
.hero {padding: 1.4rem 1.6rem; border-radius: 14px; background: linear-gradient(135deg,#0F766E 0%,#115E59 100%); color:#fff; margin-bottom:1.2rem;}
.hero h1 {margin:0; font-size:1.9rem; color:#fff;}
.hero p {margin:.4rem 0 0 0; opacity:.92;}
.result {padding: 1.1rem 1.3rem; border-radius: 12px; border-left: 8px solid var(--c); background: var(--bg);}
.result h3 {margin:0 0 .2rem 0; color: var(--c);}
.result small {color:#475569;}
.bar {display:flex; height: 34px; border-radius: 8px; overflow:hidden; margin:.7rem 0 .3rem 0; font-size:.85rem; font-weight:600; color:#fff;}
.bar div {display:flex; align-items:center; justify-content:center; min-width: 0;}
.legend {font-size:.85rem; color:#475569;}
@media (max-width: 640px) {.hero h1 {font-size:1.4rem;}}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Memuat model .pth ...")
def get_model(path: str):
    return load_checkpoint(path)


try:
    model, ck = get_model(str(MODEL_PATH))
except Exception as exc:   # tampilkan pesan yang jelas, jangan jalankan prediksi palsu
    st.error(f"Model tidak dapat dimuat dari `{MODEL_PATH}`.\n\n{exc}")
    st.stop()

NB_OPTIONS = neighbourhood_options(ck)


def default_inputs():
    today = date.today()
    return {"in_gender": "F", "in_age": 30,
            "in_nb": "CENTRO" if "CENTRO" in NB_OPTIONS else NB_OPTIONS[0],
            "in_scholarship": False, "in_hypertension": False, "in_diabetes": False, "in_alcoholism": False,
            "in_handicap": 0, "in_sms": False,
            "in_sched_date": today, "in_sched_time": time(9, 0), "in_appt_date": today + timedelta(days=7)}


def reset_inputs():
    for k, v in default_inputs().items():
        st.session_state[k] = v
    st.session_state.pop("result", None)
    st.session_state.pop("result_errors", None)


if "in_gender" not in st.session_state:
    reset_inputs()

st.markdown("""
<div class="hero">
  <h1>🩺 SmartClinic No-Show Predictor</h1>
  <p>Estimasi kemungkinan pasien rawat jalan tidak hadir pada jadwal konsultasi, menggunakan model
  PyTorch (MLP) yang dilatih pada dataset Medical Appointment No Shows.</p>
</div>
""", unsafe_allow_html=True)

tab_pred, tab_info = st.tabs(["Prediksi", "Tentang Model"])

# ------------------------------------------------------------------ TAB PREDIKSI
with tab_pred:
    st.write("Isi data pasien dan jadwal, lalu tekan **Prediksi**. Fitur *waiting_days*, *appt_weekday* dan "
             "*sched_hour* dihitung otomatis dari tanggal dan waktu, sama seperti pada Part 2.")
    col_in, col_out = st.columns([3, 2], gap="large")

    with col_in:
        st.subheader("Data pasien")
        c1, c2 = st.columns(2)
        gender_label = c1.radio("Jenis kelamin", ["Perempuan (F)", "Laki-laki (M)"], horizontal=True,
                                index=0 if st.session_state["in_gender"] == "F" else 1, key="in_gender_label")
        st.session_state["in_gender"] = "F" if gender_label.endswith("(F)") else "M"
        c2.number_input("Umur (tahun)", min_value=0, max_value=100, step=1, key="in_age",
                        help="Data latih mencakup umur 0-100 tahun.")
        st.selectbox("Kelurahan (Neighbourhood)", NB_OPTIONS, key="in_nb",
                     help="Daftar berasal dari kategori yang dikenal model. 'Lainnya' untuk kelurahan yang jarang muncul.")

        st.subheader("Kondisi dan layanan")
        d1, d2, d3 = st.columns(3)
        d1.checkbox("Penerima beasiswa (Scholarship)", key="in_scholarship",
                    help="Program bantuan sosial Bolsa Familia.")
        d2.checkbox("Hipertensi", key="in_hypertension")
        d3.checkbox("Diabetes", key="in_diabetes")
        e1, e2, e3 = st.columns(3)
        e1.checkbox("Alkoholisme", key="in_alcoholism")
        e2.checkbox("SMS pengingat terkirim", key="in_sms")
        e3.selectbox("Tingkat disabilitas (Handicap)", [0, 1, 2, 3, 4], key="in_handicap",
                     help="Skala 0-4 sesuai dataset.")

        st.subheader("Jadwal")
        f1, f2, f3 = st.columns(3)
        f1.date_input("Tanggal pendaftaran jadwal", key="in_sched_date")
        f2.time_input("Waktu pendaftaran jadwal", key="in_sched_time", step=900,
                      help="Model hanya memakai jam (sched_hour).")
        f3.date_input("Tanggal janji temu", key="in_appt_date")

        b1, b2 = st.columns(2)
        predict_clicked = b1.button("Prediksi", type="primary", use_container_width=True)
        b2.button("Hapus / ulangi input", on_click=reset_inputs, use_container_width=True)

    if predict_clicked:
        s = st.session_state
        sched_dt = datetime.combine(s["in_sched_date"], s["in_sched_time"])
        errors, warns = validate_inputs(s["in_age"], sched_dt, s["in_appt_date"], s["in_handicap"], s["in_gender"])
        if errors:
            st.session_state.pop("result", None)
            st.session_state["result_errors"] = errors
        else:
            frame = build_input_frame(s["in_gender"], s["in_age"], s["in_nb"], s["in_scholarship"],
                                      s["in_hypertension"], s["in_diabetes"], s["in_alcoholism"], s["in_handicap"],
                                      s["in_sms"], sched_dt, s["in_appt_date"])
            st.session_state["result"] = (predict_one(model, ck, frame), warns)
            st.session_state.pop("result_errors", None)

    with col_out:
        st.subheader("Hasil prediksi")
        if st.session_state.get("result_errors"):
            for msg in st.session_state["result_errors"]:
                st.error(msg)
        elif "result" in st.session_state:
            pred, warns = st.session_state["result"]
            no_show = pred.label == 1
            color, bg = ("#B45309", "#FFFBEB") if no_show else ("#0F766E", "#F0FDFA")
            st.markdown(f"""
<div class="result" style="--c:{color}; --bg:{bg};">
  <h3>{'Diperkirakan TIDAK HADIR' if no_show else 'Diperkirakan HADIR'}</h3>
  <small>Kelas model: {pred.label_text} &middot; ambang keputusan {pred.threshold:.2f}</small>
  <div class="bar">
    <div style="width:{pred.prob_show*100:.2f}%; background:#0F766E;">{pred.prob_show*100:.1f}%</div>
    <div style="width:{pred.prob_no_show*100:.2f}%; background:#D97706;">{pred.prob_no_show*100:.1f}%</div>
  </div>
  <div class="legend">&#9632; <span style="color:#0F766E">Hadir</span> &nbsp; &#9632; <span style="color:#D97706">Tidak hadir</span></div>
</div>""", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Probabilitas hadir", f"{pred.prob_show*100:.1f}%")
            m2.metric("Probabilitas tidak hadir", f"{pred.prob_no_show*100:.1f}%")
            st.caption("Model dilatih dengan pembobotan kelas (pos_weight ≈ 3,95) sehingga cenderung sensitif "
                       "terhadap pasien yang tidak hadir; nilai probabilitas bukan frekuensi sebenarnya.")
            for w in warns:
                st.warning(w)
            with st.expander("Fitur yang dikirim ke model"):
                st.dataframe(pd.DataFrame({"Fitur": list(pred.features.keys()),
                                           "Nilai": [str(v) for v in pred.features.values()]}),
                             hide_index=True, use_container_width=True)
                st.caption(f"Setelah preprocessing: vektor {ck['model_config']['input_dim']} dimensi.")
        else:
            st.info("Belum ada hasil. Lengkapi input lalu tekan **Prediksi**.")

    st.warning(DISCLAIMER)

# ------------------------------------------------------------------ TAB INFO MODEL
with tab_info:
    cfg, ti, mt = ck["model_config"], ck.get("training_info", {}), ck.get("metrics_test", {})
    st.subheader("Informasi model")
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Arsitektur", "MLP")
    i2.metric("Input", cfg["input_dim"])
    i3.metric("Hidden layer", " - ".join(str(h) for h in cfg["hidden_dims"]))
    i4.metric("Dropout", cfg["dropout"])
    st.write(f"Keluaran satu logit, probabilitas *No-show* = sigmoid(logit); ambang keputusan **{ck['threshold']}**. "
             f"Kelas 0 = **{ck['class_mapping']['0']}**, kelas 1 = **{ck['class_mapping']['1']}**.")
    if mt:
        st.subheader("Evaluasi pada data uji (dari checkpoint)")
        cols = st.columns(5)
        for c, (lab, key) in zip(cols, [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"),
                                         ("F1-score", "f1"), ("ROC-AUC", "roc_auc")]):
            if key in mt:
                c.metric(lab, f"{mt[key]:.4f}")
        if "diagnosis" in mt:
            st.caption(f"Diagnosis fit: {mt['diagnosis']} (selisih AUC latih-uji {mt.get('gap_auc', float('nan')):.4f}).")
    if COMPARISON_CSV.exists():
        st.subheader("Perbandingan Part 2 dan Part 3")
        st.dataframe(pd.read_csv(COMPARISON_CSV).round(4), hide_index=True, use_container_width=True)
    st.subheader("Data dan teknologi")
    st.markdown(f"""
- **Dataset:** {ti.get('dataset', 'Medical Appointment No Shows (Kaggle)')}; target biner *No-show* (1 = tidak hadir).
- **Fitur (12):** Gender, Age, Neighbourhood, Scholarship, Hypertension, Diabetes, Alcoholism, Handicap, SMS_received,
  waiting_days, appt_weekday, sched_hour.
- **Preprocessing:** StandardScaler (numerik), one-hot dengan penggabungan kategori jarang (kategorik), biner apa adanya;
  parameter tersimpan di dalam file `.pth`.
- **Kerangka kerja:** {ti.get('framework', 'PyTorch')}, Streamlit, pandas, NumPy. Random state: {ti.get('random_state', 42)}.
- **Part 3 vs Part 2:** Part 2 memakai model scikit-learn; Part 3 menambahkan model PyTorch pada dataset dan pipeline yang
  sama agar dapat disimpan sebagai `.pth` dan diintegrasikan ke aplikasi.
""")
    st.warning(DISCLAIMER)
