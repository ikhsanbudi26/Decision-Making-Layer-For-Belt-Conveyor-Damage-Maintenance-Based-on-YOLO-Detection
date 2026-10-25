"""
analisis_perbandingan.py - Menghasilkan confusion matrix DAN Cohen's Kappa
untuk rule-based vs regresi logistik, di DATA UJI yang SAMA -- siap
dipakai langsung untuk isi BAB 4.5 (Analisis dan Perbandingan Kinerja).

Jalankan SETELAH kalibrasi_rule_based.py & train_logreg.py selesai
(butuh severity_logreg.pkl sudah ada).

Jalankan: python3 analisis_perbandingan.py
"""

import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report

from db_severity import ambil_data_kalibrasi

KATEGORI = {"Rusak Ringan": 0, "Rusak Berat": 1}

# --- GANTI dengan hasil kalibrasi_rule_based.py kamu ---
W_R, W_N, W_C, N_MAX, TAU = 0.6, 0.2, 0.2, 3, 0.4636


def cetak_confusion_matrix(nama_metode, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"\n=== Confusion Matrix: {nama_metode} ===")
    print(f"                  Prediksi Ringan   Prediksi Berat")
    print(f"Aktual Ringan     {cm[0][0]:^17}{cm[0][1]:^17}")
    print(f"Aktual Berat      {cm[1][0]:^17}{cm[1][1]:^17}")
    print()
    print(classification_report(y_true, y_pred, target_names=["Ringan", "Berat"], digits=3))


def buat_grafik_perbandingan(y_true, y_pred_rule, y_pred_logreg):
    """Grafik batang berkelompok: Precision/Recall/F1-Score, per kelas,
    membandingkan Rule-Based vs Regresi Logistik berdampingan. Support
    TIDAK dimasukkan ke grafik ini (satuannya beda -- jumlah sesi, bukan
    skor 0-1 -- dan nilainya sama untuk kedua metode karena diuji di
    data yang sama, jadi cukup disebutkan sebagai catatan angka biasa)."""

    rep_rule = classification_report(y_true, y_pred_rule,
                                       target_names=["Ringan", "Berat"],
                                       output_dict=True, digits=3)
    rep_logreg = classification_report(y_true, y_pred_logreg,
                                         target_names=["Ringan", "Berat"],
                                         output_dict=True, digits=3)

    metrik = ["precision", "recall", "f1-score"]
    kelas = ["Ringan", "Berat"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, k in zip(axes, kelas):
        nilai_rule = [rep_rule[k][m] for m in metrik]
        nilai_logreg = [rep_logreg[k][m] for m in metrik]

        x = np.arange(len(metrik))
        lebar = 0.35

        ax.bar(x - lebar/2, nilai_rule, lebar, label="Rule-Based", color="#2563eb")
        ax.bar(x + lebar/2, nilai_logreg, lebar, label="Regresi Logistik", color="#dc2626")

        ax.set_xticks(x)
        ax.set_xticklabels(["Precision", "Recall", "F1-Score"])
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Kelas: {k} (support={rep_rule[k]['support']:.0f})")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Label angka di atas tiap batang
        for xi, v in zip(x - lebar/2, nilai_rule):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
        for xi, v in zip(x + lebar/2, nilai_logreg):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig("perbandingan_metode.png", dpi=300, bbox_inches="tight")
    print("\nGrafik tersimpan: perbandingan_metode.png")


def main():
    data = ambil_data_kalibrasi()
    valid = [d for d in data if d["category_teknisi"] in KATEGORI and d["n_detections"] >= 1]
    print(f"Total sesi valid: {len(valid)}")

    labels_all = [KATEGORI[d["category_teknisi"]] for d in valid]
    idx = list(range(len(valid)))
    idx_tr, idx_te = train_test_split(idx, test_size=0.3, stratify=labels_all, random_state=42)
    test_set = [valid[i] for i in idx_te]

    y_true = [KATEGORI[d["category_teknisi"]] for d in test_set]

    # --- Prediksi rule-based ---
    y_pred_rule = []
    for d in test_set:
        n_norm = min(d["n_detections"] / N_MAX, 1.0)
        s = W_R * d["r_max"] + W_N * n_norm + W_C * d["c_avg"]
        y_pred_rule.append(1 if s > TAU else 0)

    kappa_rule = cohen_kappa_score(y_true, y_pred_rule, weights="quadratic")
    cetak_confusion_matrix("Rule-Based", y_true, y_pred_rule)
    print(f"Cohen's Kappa Rule-Based: {kappa_rule:.3f}")

    # --- Prediksi regresi logistik ---
    model = joblib.load("severity_logreg.pkl")
    X_test = np.array([[d["r_max"], d["n_norm"], d["c_avg"]] for d in test_set])
    y_pred_logreg = model.predict(X_test)

    kappa_logreg = cohen_kappa_score(y_true, y_pred_logreg, weights="quadratic")
    cetak_confusion_matrix("Regresi Logistik", y_true, y_pred_logreg)
    print(f"Cohen's Kappa Regresi Logistik: {kappa_logreg:.3f}")

    # --- Ringkasan akhir untuk tabel di laporan ---
    print("\n" + "=" * 50)
    print("TABEL RINGKASAN UNTUK BAB 4.5.4")
    print("=" * 50)
    print(f"{'Metode':<20}{'Cohen Kappa':<15}{'Interpretasi'}")
    for nama, k in [("Rule-Based", kappa_rule), ("Regresi Logistik", kappa_logreg)]:
        if k < 0.20: interp = "Sangat lemah"
        elif k < 0.40: interp = "Lemah"
        elif k < 0.60: interp = "Sedang"
        elif k < 0.80: interp = "Kuat"
        else: interp = "Hampir sempurna"
        print(f"{nama:<20}{k:<15.3f}{interp}")

    # --- Buat grafik batang perbandingan precision/recall/f1-score ---
    buat_grafik_perbandingan(y_true, y_pred_rule, y_pred_logreg)


if __name__ == "__main__":
    main()
