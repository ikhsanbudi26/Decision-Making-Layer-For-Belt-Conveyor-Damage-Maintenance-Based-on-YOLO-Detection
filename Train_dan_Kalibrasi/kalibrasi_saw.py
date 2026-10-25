import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
from db_severity import ambil_data_kalibrasi

# === ISI DENGAN BOBOT MANUALMU ===
W_R, W_N, W_C = 0.4, 0.3, 0.3
N_MAX, TAU = 5, 0.4636

data = ambil_data_kalibrasi()
dv = [d for d in data if d["category_teknisi"] in ("Rusak Ringan", "Rusak Berat")
      and d.get("n_detections", 0) >= 1 and d["r_max"] is not None]
y = [1 if d["category_teknisi"] == "Rusak Berat" else 0 for d in dv]
idx_tr, idx_te = train_test_split(range(len(dv)), test_size=0.3, stratify=y, random_state=42)

def evaluasi(indices, nama):
    yt = np.array([y[i] for i in indices])
    skor = np.array([W_R*dv[i]["r_max"] + W_N*min(dv[i]["n_detections"]/N_MAX, 1.0)
                     + W_C*dv[i]["c_avg"] for i in indices])
    yp = (skor > TAU).astype(int)
    print(f"\n=== {nama} (n={len(yt)}) ===")
    print(confusion_matrix(yt, yp))
    print(classification_report(yt, yp, target_names=["Ringan", "Berat"], digits=3))
    print("Cohen's Kappa:", round(cohen_kappa_score(yt, yp), 3))

evaluasi(idx_te, "DATA UJI (pembanding adil dengan regresi logistik)")
evaluasi(idx_tr, "DATA LATIH (in-sample, hanya sebagai pembanding)")