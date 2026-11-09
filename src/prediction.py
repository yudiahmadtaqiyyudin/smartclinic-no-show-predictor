"""Pemuatan model .pth, validasi input, dan prediksi SmartClinic No-Show Predictor.

Semua probabilitas berasal dari model PyTorch yang dimuat dari file .pth (tanpa prediksi dummy).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .model_architecture import NoShowMLP
from .preprocessing import FEATURE_COLS, derive_time_features, transform

# Rentang data latih (Part 2 setelah pembersihan) - dipakai untuk peringatan, bukan penolakan.
TRAIN_WAITING_DAYS_MAX = 179
OTHER_NEIGHBOURHOOD = "Lainnya (kategori jarang)"
_UNKNOWN_TOKEN = "__KATEGORI_JARANG__"   # tidak ada di daftar kategori -> masuk kolom "infrequent"

REQUIRED_KEYS = ("model_state_dict", "model_config", "preprocessing", "feature_columns",
                 "feature_names_out", "class_mapping", "threshold")


def load_checkpoint(path):
    """Muat checkpoint .pth dan kembalikan (model dalam mode eval, checkpoint)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File model tidak ditemukan: {path}")
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)   # jalur aman (tanpa unpickle objek)
    except Exception:
        # Cadangan untuk checkpoint buatan sendiri yang berisi tipe numpy; hanya gunakan file milik sendiri.
        ck = torch.load(path, map_location="cpu", weights_only=False)
    missing = [k for k in REQUIRED_KEYS if k not in ck]
    if missing:
        raise ValueError(f"Checkpoint tidak lengkap, kunci hilang: {missing}")
    if list(ck["feature_columns"]) != FEATURE_COLS:
        raise ValueError("Urutan fitur checkpoint tidak sesuai dengan src/preprocessing.py")
    cfg = ck["model_config"]
    if cfg["input_dim"] != len(ck["feature_names_out"]):
        raise ValueError("input_dim tidak sama dengan jumlah fitur hasil preprocessing")
    model = NoShowMLP(cfg["input_dim"], tuple(cfg["hidden_dims"]), cfg["dropout"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck


def neighbourhood_options(ck):
    """Daftar kelurahan yang dikenal model (dari checkpoint) + satu opsi untuk kategori jarang."""
    info = ck["preprocessing"]["categories"]["Neighbourhood"]
    opts = sorted(info["frequent"])
    return opts + ([OTHER_NEIGHBOURHOOD] if info["has_infrequent"] else [])


def validate_inputs(age, scheduled_dt, appointment_date, handicap, gender):
    """Kembalikan (daftar_error, daftar_peringatan). Error menghalangi prediksi."""
    errors, warnings = [], []
    if gender not in ("F", "M"):
        errors.append("Jenis kelamin harus F atau M.")
    if age is None or not (0 <= int(age) <= 100):
        errors.append("Umur harus berada pada rentang 0-100 tahun (sama dengan data latih).")
    if handicap is None or int(handicap) not in (0, 1, 2, 3, 4):
        errors.append("Nilai Handicap harus 0-4.")
    if scheduled_dt is None or appointment_date is None:
        errors.append("Tanggal pendaftaran dan tanggal janji temu wajib diisi.")
        return errors, warnings
    t = derive_time_features(scheduled_dt, appointment_date)
    if t["waiting_days"] < 0:
        errors.append("Tanggal janji temu tidak boleh sebelum tanggal pendaftaran jadwal "
                      "(data seperti ini dibuang pada Part 2).")
    elif t["waiting_days"] > TRAIN_WAITING_DAYS_MAX:
        warnings.append(f"Jeda {t['waiting_days']} hari melebihi data latih (maksimum {TRAIN_WAITING_DAYS_MAX} hari); "
                        "estimasi kurang andal.")
    if t["appt_weekday"] in ("5", "6"):
        warnings.append("Data latih hampir tidak memuat janji temu hari Sabtu/Minggu (hanya 39 dari 110.514 baris, "
                        "semuanya Sabtu); estimasi kurang andal.")
    return errors, warnings


def build_input_frame(gender, age, neighbourhood, scholarship, hypertension, diabetes, alcoholism,
                      handicap, sms_received, scheduled_dt, appointment_date) -> pd.DataFrame:
    """Susun satu baris fitur mentah (12 kolom, urutan FEATURE_COLS) dari input pengguna."""
    t = derive_time_features(scheduled_dt, appointment_date)
    nb = _UNKNOWN_TOKEN if neighbourhood == OTHER_NEIGHBOURHOOD else str(neighbourhood)
    row = {"Gender": gender, "Age": int(age), "Neighbourhood": nb, "Scholarship": int(scholarship),
           "Hypertension": int(hypertension), "Diabetes": int(diabetes), "Alcoholism": int(alcoholism),
           "Handicap": int(handicap), "SMS_received": int(sms_received), **t}
    return pd.DataFrame([row])[FEATURE_COLS]


def predict_proba(model, ck, frame: pd.DataFrame) -> np.ndarray:
    """Probabilitas No-show (kelas 1) untuk setiap baris, langsung dari model .pth."""
    x = torch.as_tensor(transform(frame, ck["preprocessing"]))
    with torch.inference_mode():
        return torch.sigmoid(model(x)).numpy()


@dataclass
class Prediction:
    prob_no_show: float
    prob_show: float
    label: int
    label_text: str
    threshold: float
    features: dict


def predict_one(model, ck, frame: pd.DataFrame) -> Prediction:
    p = float(predict_proba(model, ck, frame)[0])
    thr = float(ck["threshold"])
    label = int(p >= thr)
    return Prediction(prob_no_show=p, prob_show=1.0 - p, label=label,
                      label_text=ck["class_mapping"][str(label)], threshold=thr,
                      features=frame.iloc[0].to_dict())
