"""Verifikasi: model .pth dapat dimuat ulang (weights_only=True) dan menghasilkan prediksi konsisten."""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # path relatif terhadap folder proyek
import json, os, sys
import numpy as np, torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from src.preprocessing import RANDOM_STATE, load_and_clean, transform
from src.model_architecture import NoShowMLP

PATH = sys.argv[1] if len(sys.argv) > 1 else "model/model_no_show.pth"
DATA = sys.argv[2] if len(sys.argv) > 2 else "data/KaggleV2-May-2016.csv"

def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)   # aman: tanpa unpickle objek arbitrer
    cfg = ck["model_config"]
    m = NoShowMLP(cfg["input_dim"], tuple(cfg["hidden_dims"]), cfg["dropout"])
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m, ck

def proba(m, X):
    with torch.inference_mode():
        return torch.sigmoid(m(torch.as_tensor(X))).numpy()

print(f"File: {PATH} | ukuran {os.path.getsize(PATH)/1024:.1f} KB")
m1, ck = load(PATH); m2, _ = load(PATH)
print("Kunci checkpoint:", list(ck.keys()))
print("Konfigurasi model:", ck["model_config"], "| jumlah fitur:", len(ck["feature_names_out"]))
print("Pemetaan kelas:", ck["class_mapping"], "| threshold:", ck["threshold"])

X, y = load_and_clean(DATA)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
Xt = transform(X_test, ck["preprocessing"])
p1, p2 = proba(m1, Xt), proba(m2, Xt)
auc = roc_auc_score(y_test, p1); f1 = f1_score(y_test, (p1 >= ck["threshold"]).astype(int))
print(f"Dua kali muat -> selisih prediksi maks: {np.abs(p1-p2).max():.2e}")
print(f"ROC-AUC uji (dihitung ulang dari .pth): {auc:.4f} | tersimpan: {ck['metrics_test']['roc_auc']:.4f}")
print(f"F1 uji (dihitung ulang dari .pth): {f1:.4f} | tersimpan: {ck['metrics_test']['f1']:.4f}")
assert np.abs(p1-p2).max() < 1e-7
assert abs(auc - ck["metrics_test"]["roc_auc"]) < 1e-6 and abs(f1 - ck["metrics_test"]["f1"]) < 1e-6
assert not np.isnan(p1).any() and ((p1 >= 0) & (p1 <= 1)).all()

# Satu contoh baris mentah -> prediksi (jalur yang sama dengan aplikasi)
row = X_test.iloc[[0]]
pr = float(proba(m1, transform(row, ck["preprocessing"]))[0])
print("Contoh baris:", row.iloc[0].to_dict())
print(f"Probabilitas no-show: {pr:.4f} -> {ck['class_mapping'][str(int(pr >= ck['threshold']))]} | label asli: {int(y_test.iloc[0])}")
print("VERIFIKASI BERHASIL")
