"""Pengujian pemuatan model .pth, preprocessing, validasi input, dan prediksi."""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.prediction import (OTHER_NEIGHBOURHOOD, build_input_frame, load_checkpoint, neighbourhood_options,
                            predict_one, predict_proba, validate_inputs)
from src.preprocessing import FEATURE_COLS, derive_time_features, feature_names, transform

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model" / "model_no_show.pth"
DATA = ROOT / "data" / "KaggleV2-May-2016.csv"


@pytest.fixture(scope="module")
def loaded():
    return load_checkpoint(MODEL)


def sample(**over):
    base = dict(gender="F", age=35, neighbourhood="CENTRO", scholarship=0, hypertension=0, diabetes=0,
                alcoholism=0, handicap=0, sms_received=1, scheduled_dt=datetime(2026, 11, 2, 10, 30),
                appointment_date=datetime(2026, 11, 9))
    base.update(over)
    return build_input_frame(**base)


def test_checkpoint_sesuai_spesifikasi(loaded):
    model, ck = loaded
    assert ck["model_config"]["input_dim"] == len(ck["feature_names_out"])
    assert ck["threshold"] == 0.5
    assert ck["class_mapping"] == {"0": "Hadir", "1": "Tidak hadir (No-show)"}
    assert ck["feature_columns"] == FEATURE_COLS and len(FEATURE_COLS) == 12
    assert not model.training            # mode eval


def test_model_ini_92_fitur_64_32(loaded):
    _, ck = loaded
    cfg = ck["model_config"]
    assert (cfg["input_dim"], cfg["hidden_dims"], cfg["dropout"]) == (92, [64, 32], 0.3)


def test_nama_fitur_sama_dengan_parameter(loaded):
    _, ck = loaded
    assert feature_names(ck["preprocessing"]) == ck["feature_names_out"]


def test_fitur_waktu_sesuai_part2():
    t = derive_time_features(datetime(2016, 4, 29, 18, 38, 8), datetime(2016, 5, 3))
    assert t == {"waiting_days": 4, "appt_weekday": "1", "sched_hour": 18}   # 3 Mei 2016 = Selasa
    assert derive_time_features("2026-11-09 08:00", "2026-11-09")["waiting_days"] == 0


def test_transform_bentuk_dan_one_hot(loaded):
    _, ck = loaded
    x = transform(sample(), ck["preprocessing"])
    assert x.shape == (1, 92) and x.dtype == np.float32
    names = ck["feature_names_out"]
    row = dict(zip(names, x[0]))
    assert row["Gender=F"] == 1 and row["Gender=M"] == 0
    assert row["Neighbourhood=CENTRO"] == 1 and sum(v for k, v in row.items() if k.startswith("Neighbourhood=")) == 1
    assert row["appt_weekday=0"] == 1                      # 9 Nov 2026 = Senin
    assert row["SMS_received"] == 1 and row["Scholarship"] == 0
    assert row["sched_hour"] == pytest.approx((10 - ck["preprocessing"]["num_mean"][2]) / ck["preprocessing"]["num_scale"][2])


def test_kategori_jarang_masuk_kolom_infrequent(loaded):
    _, ck = loaded
    names = ck["feature_names_out"]
    x = transform(sample(neighbourhood=OTHER_NEIGHBOURHOOD, appointment_date=datetime(2026, 11, 14)), ck["preprocessing"])
    row = dict(zip(names, x[0]))
    assert row["Neighbourhood=infrequent"] == 1 and row["appt_weekday=infrequent"] == 1   # Sabtu


def test_prediksi_valid_dan_cocok_dengan_output_model(loaded):
    model, ck = loaded
    frame = sample()
    pred = predict_one(model, ck, frame)
    assert 0.0 <= pred.prob_no_show <= 1.0
    assert pred.prob_show + pred.prob_no_show == pytest.approx(1.0)
    assert pred.label == int(pred.prob_no_show >= 0.5) and pred.label_text == ck["class_mapping"][str(pred.label)]
    with torch.no_grad():                                   # bandingkan dengan forward manual
        manual = torch.sigmoid(model(torch.as_tensor(transform(frame, ck["preprocessing"])))).item()
    assert pred.prob_no_show == pytest.approx(manual, abs=1e-7)
    assert predict_one(model, ck, frame).prob_no_show == pred.prob_no_show      # deterministik


def test_prediksi_tidak_statis(loaded):
    model, ck = loaded
    p = {predict_one(model, ck, f).prob_no_show for f in
         [sample(age=3), sample(age=80, sms_received=0), sample(appointment_date=datetime(2026, 12, 20))]}
    assert len(p) == 3


def test_muat_ulang_konsisten(loaded):
    model, ck = loaded
    model2, ck2 = load_checkpoint(MODEL)
    f = sample()
    assert np.array_equal(predict_proba(model, ck, f), predict_proba(model2, ck2, f))


def test_opsi_kelurahan(loaded):
    _, ck = loaded
    opts = neighbourhood_options(ck)
    assert "CENTRO" in opts and opts[-1] == OTHER_NEIGHBOURHOOD and len(opts) == 75


@pytest.mark.parametrize("kwargs, fragment", [
    (dict(age=-1), "Umur"), (dict(age=101), "Umur"), (dict(handicap=7), "Handicap"), (dict(gender="X"), "kelamin"),
    (dict(appointment_date=datetime(2026, 11, 1)), "sebelum"),
])
def test_input_tidak_valid_ditolak(kwargs, fragment):
    args = dict(age=35, scheduled_dt=datetime(2026, 11, 2, 10), appointment_date=datetime(2026, 11, 9),
                handicap=0, gender="F")
    args.update(kwargs)
    errors, _ = validate_inputs(**args)
    assert errors and any(fragment in e for e in errors)


def test_peringatan_di_luar_data_latih():
    _, warns = validate_inputs(35, datetime(2026, 1, 1, 9), datetime(2026, 11, 9), 0, "F")      # jeda > 179 hari
    assert any("179" in w for w in warns)
    _, warns = validate_inputs(35, datetime(2026, 11, 9, 9), datetime(2026, 11, 15), 0, "F")    # Minggu
    assert any("Sabtu/Minggu" in w for w in warns)


def test_checkpoint_rusak_ditolak(tmp_path):
    bad = tmp_path / "bad.pth"
    torch.save({"model_state_dict": {}}, bad)
    with pytest.raises(ValueError):
        load_checkpoint(bad)
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "tidak_ada.pth")


@pytest.mark.skipif(not DATA.exists(), reason="CSV Kaggle tidak ada di data/ (opsional)")
def test_metrik_uji_direproduksi_dari_pth(loaded):
    """Menghitung ulang ROC-AUC dan F1 data uji dari .pth; harus sama dengan nilai tersimpan di checkpoint."""
    pytest.importorskip("sklearn")
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from src.preprocessing import RANDOM_STATE, load_and_clean
    model, ck = loaded
    X, y = load_and_clean(DATA)
    _, Xte, _, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    p = predict_proba(model, ck, Xte)
    assert roc_auc_score(yte, p) == pytest.approx(ck["metrics_test"]["roc_auc"], abs=1e-6)
    assert f1_score(yte, (p >= ck["threshold"]).astype(int)) == pytest.approx(ck["metrics_test"]["f1"], abs=1e-6)
