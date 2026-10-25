"""
train_logreg.py - Melatih model regresi logistik dari data kalibrasi
yang sudah diberi label teknisi, lalu menyimpannya sebagai file .pkl
untuk dipakai main_app.py.

WAJIB dijalankan SETELAH cukup banyak sesi (idealnya 60-80) sudah
diberi label lewat update_label_teknisi() di db_severity.py.

Jalankan: python3 train_logreg.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, cohen_kappa_score
import joblib

from db_severity import ambil_data_kalibrasi

MODEL_OUTPUT_PATH = "severity_logreg.pkl"


def main():
    data = ambil_data_kalibrasi()
    print(f"Total sesi berlabel teknisi ditemukan: {len(data)}")

    # Buang sesi "Tidak Rusak" -- regresi logistik cuma bedakan
    # Ringan vs Berat, karena Tidak Rusak sudah ditangani trivial (N=0)
    data_valid = [d for d in data if d["category_teknisi"] in ("Rusak Ringan", "Rusak Berat")
                  and d["r_max"] is not None]
    print(f"Sesi valid untuk training (Ringan/Berat saja): {len(data_valid)}")

    if len(data_valid) < 15:
        print("\n⚠️  Data masih terlalu sedikit untuk training yang meaningful.")
        print("   Disarankan kumpulkan minimal 30-40 sesi berlabel dulu.")
        print("   Program dihentikan, TIDAK membuat model dengan data sesedikit ini.")
        return

    X = np.array([[d["r_max"], d["n_norm"], d["c_avg"]] for d in data_valid])
    y = np.array([1 if d["category_teknisi"] == "Rusak Berat" else 0 for d in data_valid])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    print("\n=== KOEFISIEN HASIL TRAINING ===")
    print(f"Intercept (β0): {model.intercept_[0]:.4f}")
    print(f"β1 (R_max)    : {model.coef_[0][0]:.4f}")
    print(f"β2 (N_norm)   : {model.coef_[0][1]:.4f}")
    print(f"β3 (C_avg)    : {model.coef_[0][2]:.4f}")

    y_pred = model.predict(X_test)
    print("\n=== EVALUASI DI DATA UJI ===")
    print(classification_report(y_test, y_pred, target_names=["Ringan", "Berat"]))
    kappa = cohen_kappa_score(y_test, y_pred, weights="quadratic")
    print(f"Quadratic Weighted Cohen's Kappa: {kappa:.3f}")

    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"\n✅ Model tersimpan di: {MODEL_OUTPUT_PATH}")
    print("   Restart main_app.py supaya model baru ini ikut dimuat.")


if __name__ == "__main__":
    main()
