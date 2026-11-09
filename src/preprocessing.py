"""Preprocessing SmartClinic No-Show Predictor.

Langkah pembersihan dan rekayasa fitur identik dengan notebook Part 2
(Tugas_Besar_ML_Part2.ipynb). Encoding/scaling disimpan sebagai parameter sederhana
(list/angka) di dalam file .pth sehingga aplikasi tidak bergantung pada objek scikit-learn.
"""
import numpy as np
import pandas as pd

RANDOM_STATE = 42
NUM_COLS = ["Age", "waiting_days", "sched_hour", "Handicap"]
CAT_COLS = ["Gender", "Neighbourhood", "appt_weekday"]
BIN_COLS = ["Scholarship", "Hypertension", "Diabetes", "Alcoholism", "SMS_received"]
FEATURE_COLS = ["Gender", "Age", "Neighbourhood", "Scholarship", "Hypertension", "Diabetes",
                "Alcoholism", "Handicap", "SMS_received", "waiting_days", "appt_weekday", "sched_hour"]


def derive_time_features(scheduled_day, appointment_day):
    """Hitung waiting_days, appt_weekday (0=Senin, string) dan sched_hour (sama dengan Part 2)."""
    sched = pd.Timestamp(scheduled_day)
    appt = pd.Timestamp(appointment_day)
    waiting_days = (appt.normalize() - sched.normalize()).days
    return {"waiting_days": int(waiting_days), "appt_weekday": str(appt.dayofweek),
            "sched_hour": int(sched.hour)}


def load_and_clean(path):
    """Memuat CSV Kaggle dan menerapkan pembersihan + rekayasa fitur Part 2.

    Return: X (DataFrame fitur), y (Series target; 1 = tidak hadir / No-show).
    """
    df = pd.read_csv(path)
    data = df.rename(columns={"Hipertension": "Hypertension", "Handcap": "Handicap", "No-show": "No_show"})
    data["ScheduledDay"] = pd.to_datetime(data["ScheduledDay"], utc=True).dt.tz_localize(None)
    data["AppointmentDay"] = pd.to_datetime(data["AppointmentDay"], utc=True).dt.tz_localize(None)
    data["waiting_days"] = (data["AppointmentDay"].dt.normalize() - data["ScheduledDay"].dt.normalize()).dt.days
    data["appt_weekday"] = data["AppointmentDay"].dt.dayofweek.astype(str)
    data["sched_hour"] = data["ScheduledDay"].dt.hour
    data = data[(data["Age"] >= 0) & (data["Age"] <= 100) & (data["waiting_days"] >= 0)]
    data["target"] = (data["No_show"] == "Yes").astype(int)
    data = data.drop(columns=["PatientId", "AppointmentID", "ScheduledDay", "AppointmentDay", "No_show"])
    data["Handicap"] = data["Handicap"].astype(int)
    return data.drop(columns="target")[FEATURE_COLS], data["target"]


def build_sklearn_preprocessor():
    """ColumnTransformer yang sama persis dengan Part 2 (dipakai saat training)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100), CAT_COLS),
        ("bin", "passthrough", BIN_COLS),
    ])


def extract_params(fitted_ct):
    """Ambil parameter scaler dan encoder dari ColumnTransformer yang sudah di-fit."""
    scaler = fitted_ct.named_transformers_["num"]
    ohe = fitted_ct.named_transformers_["cat"]
    infreq = ohe.infrequent_categories_
    cat_params = {}
    for i, col in enumerate(CAT_COLS):
        cats = [str(c) for c in ohe.categories_[i]]
        infrequent = [str(c) for c in infreq[i]] if infreq[i] is not None else []
        cat_params[col] = {"frequent": [c for c in cats if c not in infrequent],
                           "has_infrequent": len(infrequent) > 0}
    return {
        "num_cols": list(NUM_COLS), "cat_cols": list(CAT_COLS), "bin_cols": list(BIN_COLS),
        "num_mean": [float(v) for v in scaler.mean_], "num_scale": [float(v) for v in scaler.scale_],
        "categories": cat_params,
    }


def feature_names(params):
    """Nama kolom setelah preprocessing (urutan sama dengan vektor input model)."""
    names = list(params["num_cols"])
    for col in params["cat_cols"]:
        info = params["categories"][col]
        names += [f"{col}={c}" for c in info["frequent"]]
        if info["has_infrequent"]:
            names.append(f"{col}=infrequent")
    return names + list(params["bin_cols"])


def transform(df, params):
    """Ubah DataFrame fitur mentah menjadi matriks float32 untuk model PyTorch."""
    n = len(df)
    num = df[params["num_cols"]].to_numpy(dtype=np.float64)
    num = (num - np.array(params["num_mean"])) / np.array(params["num_scale"])
    blocks = [num]
    for col in params["cat_cols"]:
        info = params["categories"][col]
        values = df[col].astype(str).to_numpy()
        freq = info["frequent"]
        m = np.zeros((n, len(freq) + (1 if info["has_infrequent"] else 0)))
        for j, c in enumerate(freq):
            m[:, j] = values == c
        if info["has_infrequent"]:
            m[:, -1] = ~np.isin(values, freq)   # kategori jarang/tidak dikenal
        blocks.append(m)
    blocks.append(df[params["bin_cols"]].to_numpy(dtype=np.float64))
    return np.hstack(blocks).astype(np.float32)
