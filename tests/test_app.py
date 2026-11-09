"""Pengujian antarmuka Streamlit tanpa browser (streamlit.testing.AppTest)."""
from datetime import date, time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def run_app():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    return at


def button(at, label):
    return next(b for b in at.button if label in b.label)


def test_halaman_awal_tanpa_hasil():
    at = run_app()
    assert "SmartClinic" in at.markdown[1].value or any("SmartClinic" in m.value for m in at.markdown)
    assert any("Belum ada hasil" in i.value for i in at.info)


def test_prediksi_menampilkan_hasil_dari_model():
    at = run_app()
    at.number_input(key="in_age").set_value(60)
    at.date_input(key="in_sched_date").set_value(date(2026, 11, 2))
    at.date_input(key="in_appt_date").set_value(date(2026, 11, 16))
    button(at, "Prediksi").click(); at.run()
    assert not at.exception
    pred, _ = at.session_state["result"]
    assert 0 <= pred.prob_no_show <= 1 and pred.features["waiting_days"] == 14 and pred.features["Age"] == 60
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Probabilitas hadir"].endswith("%") and metrics["Probabilitas tidak hadir"].endswith("%")
    assert any("proyek akademik" in w.value for w in at.warning)


def test_tanggal_janji_sebelum_pendaftaran_ditolak():
    at = run_app()
    at.date_input(key="in_sched_date").set_value(date(2026, 11, 10))
    at.date_input(key="in_appt_date").set_value(date(2026, 11, 5))
    button(at, "Prediksi").click(); at.run()
    assert "result" not in at.session_state and any("sebelum" in e.value for e in at.error)


def test_tombol_hapus_mengembalikan_default():
    at = run_app()
    at.number_input(key="in_age").set_value(77)
    button(at, "Prediksi").click(); at.run()
    assert "result" in at.session_state
    button(at, "Hapus").click(); at.run()
    assert "result" not in at.session_state
    assert at.number_input(key="in_age").value == 30
    assert any("Belum ada hasil" in i.value for i in at.info)


def test_input_berbeda_menghasilkan_probabilitas_berbeda():
    at = run_app()
    button(at, "Prediksi").click(); at.run()
    p1 = at.session_state["result"][0].prob_no_show
    at.number_input(key="in_age").set_value(5); at.checkbox(key="in_sms").set_value(True)
    button(at, "Prediksi").click(); at.run()
    assert at.session_state["result"][0].prob_no_show != p1
