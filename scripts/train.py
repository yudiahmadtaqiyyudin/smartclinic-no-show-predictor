"""Training model PyTorch (Opsi A - portasi Part 2) + evaluasi + perbandingan dengan Part 2.

Pipeline identik dengan Part 2: pembersihan data yang sama, stratified split 80/20 dan
random_state=42, ColumnTransformer yang sama (di-fit hanya pada data latih).
Pemilihan konfigurasi MLP memakai validation set yang diambil dari data LATIH (10%),
sehingga data uji hanya dipakai untuk evaluasi akhir.

Pemakaian:
    python scripts/train.py --data data/KaggleV2-May-2016.csv --baseline reports/evaluation/baseline_part2.json
"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # path relatif terhadap folder proyek
import argparse, json, os, random, time
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, average_precision_score)

from src.preprocessing import (RANDOM_STATE, NUM_COLS, CAT_COLS, BIN_COLS, FEATURE_COLS, load_and_clean,
                               build_sklearn_preprocessor, extract_params, feature_names, transform)
from src.model_architecture import NoShowMLP

CONFIGS = [  # ruang pencarian kecil (mirip tuning Part 2, tetapi untuk MLP)
    {"hidden_dims": (32,),     "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-5},
    {"hidden_dims": (64, 32),  "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-5},
    {"hidden_dims": (64, 32),  "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-3},
    {"hidden_dims": (128, 64), "dropout": 0.4, "lr": 1e-3, "weight_decay": 1e-4},
    {"hidden_dims": (128, 64), "dropout": 0.5, "lr": 5e-4, "weight_decay": 1e-3},
    {"hidden_dims": (32, 16),  "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4},
]
MAX_EPOCHS, PATIENCE, BATCH = 40, 6, 512
THRESHOLD = 0.5


def set_seed(seed=RANDOM_STATE):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(X))).numpy()


def metrics(y_true, prob, thr=THRESHOLD):
    pred = (prob >= thr).astype(int)
    return {"accuracy": float(accuracy_score(y_true, pred)), "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred)), "f1": float(f1_score(y_true, pred)),
            "roc_auc": float(roc_auc_score(y_true, prob)), "pr_auc": float(average_precision_score(y_true, prob)),
            "confusion_matrix": confusion_matrix(y_true, pred).tolist()}


def fit_one(cfg, Xtr, ytr, Xva, yva, pos_weight, verbose=False):
    set_seed()
    model = NoShowMLP(Xtr.shape[1], cfg["hidden_dims"], cfg["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
    Xt, yt = torch.as_tensor(Xtr), torch.as_tensor(ytr, dtype=torch.float32)
    Xv, yv = torch.as_tensor(Xva), torch.as_tensor(yva, dtype=torch.float32)
    g = torch.Generator().manual_seed(RANDOM_STATE)
    best = {"val_loss": 1e9, "state": None, "epoch": 0}
    history, wait = [], 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        tr_loss = 0.0
        for i in range(0, len(Xt), BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2:      # BatchNorm butuh >1 sampel
                continue
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward(); opt.step()
            tr_loss += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        history.append({"epoch": epoch, "train_loss": tr_loss / len(Xt), "val_loss": val_loss})
        if verbose:
            print(f"  epoch {epoch:02d} train_loss={tr_loss/len(Xt):.4f} val_loss={val_loss:.4f}")
        if val_loss < best["val_loss"] - 1e-5:
            best = {"val_loss": val_loss, "state": {k: v.clone() for k, v in model.state_dict().items()}, "epoch": epoch}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best["state"])
    return model, best["epoch"], history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/KaggleV2-May-2016.csv")
    ap.add_argument("--baseline", default="reports/evaluation/baseline_part2.json")
    ap.add_argument("--out-model", default="model/model_no_show.pth")
    ap.add_argument("--reports", default="reports/evaluation")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_model), exist_ok=True); os.makedirs(args.reports, exist_ok=True)
    t0 = time.time()

    # 1) Data + split IDENTIK dengan Part 2
    X, y = load_and_clean(args.data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print(f"Data bersih: {X.shape} | train {X_train.shape} | test {X_test.shape}")

    # 2) Preprocessing (fit hanya pada data latih), lalu validasi ekuivalensi dengan transform manual
    ct = build_sklearn_preprocessor().fit(X_train)
    params = extract_params(ct)
    names = feature_names(params)
    Xtr_all = transform(X_train, params); Xte = transform(X_test, params)
    ref = ct.transform(X_test); ref = ref.toarray() if hasattr(ref, "toarray") else np.asarray(ref)
    max_diff = float(np.abs(ref.astype(np.float32) - Xte).max())
    print(f"Dimensi fitur: {Xtr_all.shape[1]} | selisih maks transform manual vs scikit-learn: {max_diff:.2e}")
    assert Xtr_all.shape[1] == len(names) and max_diff < 1e-4, "Preprocessing manual tidak ekuivalen!"

    # 3) Validation set (10% dari data LATIH) untuk early stopping dan pemilihan konfigurasi
    ytr_all = y_train.to_numpy()
    Xtr, Xva, ytr, yva = train_test_split(Xtr_all, ytr_all, test_size=0.1, stratify=ytr_all, random_state=RANDOM_STATE)
    pos_weight = float((ytr == 0).sum() / (ytr == 1).sum())   # setara class_weight="balanced"
    print(f"pos_weight = {pos_weight:.3f} | train {Xtr.shape[0]} | val {Xva.shape[0]}")

    # 4) Tuning konfigurasi (kriteria: F1 kelas no-show pada validation, sama dengan Part 2)
    rows, best_i, best_f1 = [], 0, -1
    for i, cfg in enumerate(CONFIGS):
        model, ep, hist = fit_one(cfg, Xtr, ytr, Xva, yva, pos_weight)
        pva = predict_proba(model, Xva); m = metrics(yva, pva)
        rows.append({"config": {**cfg, "hidden_dims": list(cfg["hidden_dims"])}, "best_epoch": ep,
                     "val_f1": m["f1"], "val_roc_auc": m["roc_auc"], "val_loss": min(h["val_loss"] for h in hist)})
        print(f"[config {i}] {rows[-1]['config']} epoch={ep} val_F1={m['f1']:.4f} val_AUC={m['roc_auc']:.4f}")
        if m["f1"] > best_f1:
            best_f1, best_i, best_model, best_hist, best_ep = m["f1"], i, model, hist, ep
    cfg = CONFIGS[best_i]
    print(f"\nKonfigurasi terpilih: #{best_i} {cfg} (epoch terbaik {best_ep})")

    # 5) Evaluasi akhir pada data uji + diagnosis fit (fungsi diagnose sama dengan Part 2)
    p_tr = predict_proba(best_model, Xtr_all); p_te = predict_proba(best_model, Xte)
    res = metrics(y_test.to_numpy(), p_te)
    train_auc = float(roc_auc_score(ytr_all, p_tr)); gap = train_auc - res["roc_auc"]
    diagnosis = "Overfit" if gap > 0.05 else ("Underfit" if train_auc < 0.65 else "Good fit")
    res.update({"train_auc": train_auc, "gap_auc": gap, "diagnosis": diagnosis})
    print("Uji PyTorch:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()})

    # 6) Simpan .pth (state_dict + konfigurasi + preprocessing + pemetaan kelas; aman utk weights_only=True)
    checkpoint = {
        "model_state_dict": best_model.state_dict(),
        "model_config": {"input_dim": int(Xtr.shape[1]), "hidden_dims": list(cfg["hidden_dims"]), "dropout": cfg["dropout"]},
        "preprocessing": params,
        "feature_columns": FEATURE_COLS,
        "feature_names_out": names,
        "class_mapping": {"0": "Hadir", "1": "Tidak hadir (No-show)"},
        "threshold": THRESHOLD,
        "metrics_test": {k: v for k, v in res.items() if k != "confusion_matrix"},
        "training_info": {"framework": f"PyTorch {torch.__version__}", "random_state": RANDOM_STATE,
                          "dataset": "Medical Appointment No Shows (Kaggle)", "best_epoch": best_ep,
                          "pos_weight": pos_weight, "batch_size": BATCH, "optimizer": "Adam"},
    }
    torch.save(checkpoint, args.out_model)
    print(f"Model tersimpan: {args.out_model} ({os.path.getsize(args.out_model)/1024:.1f} KB)")

    # 7) Perbandingan dengan Part 2 (baseline di-fit ulang dengan hyperparameter terbaik hasil tuning Part 2)
    comp = {"PyTorch MLP (Part 3)": {k: v for k, v in res.items() if k in
            ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "train_auc", "diagnosis", "confusion_matrix")}}
    if os.path.exists(args.baseline):
        base = json.load(open(args.baseline))
        for name, b in base.items():
            comp[f"{name} (Part 2)"] = {k: b.get(k) for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                                                              "train_auc", "diagnosis", "confusion_matrix")}
    json.dump({"pytorch": res, "tuning": rows, "comparison": comp, "history": best_hist, "n_features": int(Xtr.shape[1]),
               "split": {"train": int(len(X_train)), "test": int(len(X_test)), "fit": int(len(ytr)), "val": int(len(yva))},
               "preprocessing_max_diff": max_diff},
              open(os.path.join(args.reports, "metrics_pytorch.json"), "w"), indent=2, default=float)
    print(f"Selesai dalam {time.time()-t0:.1f} detik")


if __name__ == "__main__":
    main()
