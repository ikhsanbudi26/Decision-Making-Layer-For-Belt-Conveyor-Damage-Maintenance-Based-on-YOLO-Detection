"""
kalibrasi_rule_based.py - Mencari kombinasi w_R, w_N, w_C, N_max, tau
yang paling sepakat dengan penilaian teknisi, lewat grid search
(untuk bobot & N_max) + optimasi threshold eksak (untuk tau).

Data dipisah 70% training (cari parameter) / 30% uji (validasi jujur),
supaya sebanding dengan cara train_logreg.py dievaluasi.

Jalankan SETELAH cukup sesi (min. 30, idealnya 60-80) diberi label teknisi.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

from db_severity import ambil_data_kalibrasi

KATEGORI_KE_ANGKA = {"Rusak Ringan": 0, "Rusak Berat": 1}


def cari_tau_optimal(scores, labels):
    """Cari threshold tau yang memaksimalkan Cohen's Kappa, dengan mencoba
    titik tengah di antara tiap skor unik yang berurutan -- metode eksak,
    bukan grid kasar yang bisa melewatkan titik optimal sesungguhnya."""
    scores = np.array(scores)
    labels = np.array(labels)
    nilai_unik = sorted(set(scores))

    titik_uji = [nilai_unik[0] - 0.01]
    for i in range(len(nilai_unik) - 1):
        titik_uji.append((nilai_unik[i] + nilai_unik[i + 1]) / 2)
    titik_uji.append(nilai_unik[-1] + 0.01)

    tau_terbaik, kappa_terbaik = titik_uji[0], -1
    for tau in titik_uji:
        pred = (scores > tau).astype(int)
        k = cohen_kappa_score(labels, pred)
        if k > kappa_terbaik:
            kappa_terbaik, tau_terbaik = k, tau
    return tau_terbaik, kappa_terbaik


def hitung_kappa_untuk_parameter(w_r, w_n, w_c, n_max, tau,
                                  r_max_list, n_list, c_avg_list, labels):
    scores = [w_r * r_max + w_n * min(n / n_max, 1.0) + w_c * c_avg
              for r_max, n, c_avg in zip(r_max_list, n_list, c_avg_list)]
    pred = (np.array(scores) > tau).astype(int)
    return cohen_kappa_score(labels, pred)


def main():
    data = ambil_data_kalibrasi()
    data_valid = [d for d in data if d["category_teknisi"] in KATEGORI_KE_ANGKA
                  and d.get("n_detections", 0) >= 1 and d["r_max"] is not None]
    print(f"Sesi valid untuk kalibrasi (Ringan/Berat saja): {len(data_valid)}")

    if len(data_valid) < 15:
        print("Data terlalu sedikit untuk kalibrasi yang meaningful.")
        print("Disarankan kumpulkan minimal 30-40 sesi berlabel dulu.")
        return

    labels_semua = [KATEGORI_KE_ANGKA[d["category_teknisi"]] for d in data_valid]

    # --- Split 70% train / 30% test, stratified spy proporsi kelas terjaga ---
    idx = list(range(len(data_valid)))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.3, stratify=labels_semua, random_state=42
    )

    def ambil_subset(indices):
        subset = [data_valid[i] for i in indices]
        r_max = [d["r_max"] for d in subset]
        n = [d["n_detections"] for d in subset]
        c_avg = [d["c_avg"] for d in subset]
        y = [KATEGORI_KE_ANGKA[d["category_teknisi"]] for d in subset]
        return r_max, n, c_avg, y

    r_max_tr, n_tr, c_avg_tr, y_tr = ambil_subset(idx_train)
    r_max_te, n_te, c_avg_te, y_te = ambil_subset(idx_test)

    print(f"Data training: {len(idx_train)} sesi | Data uji: {len(idx_test)} sesi")

    # --- Grid search bobot (w_R, w_N -> w_C otomatis) & N_max, PAKAI DATA TRAINING ---
    hasil_terbaik = {"kappa_train": -1}
    for w_r in np.arange(0.1, 0.85, 0.1):
        for w_n in np.arange(0.1, 0.85 - w_r, 0.1):
            w_c = round(1.0 - w_r - w_n, 4)
            if w_c < 0.05:
                continue
            for n_max in [3, 4, 5, 6, 7, 8, 10]:
                scores_tr = [w_r * r + w_n * min(n / n_max, 1.0) + w_c * c
                             for r, n, c in zip(r_max_tr, n_tr, c_avg_tr)]
                tau, kappa_tr = cari_tau_optimal(scores_tr, y_tr)

                if kappa_tr > hasil_terbaik["kappa_train"]:
                    hasil_terbaik = {
                        "w_r": round(w_r, 2), "w_n": round(w_n, 2), "w_c": round(w_c, 2),
                        "n_max": n_max, "tau": round(tau, 4), "kappa_train": round(kappa_tr, 4),
                    }

    # --- Validasi jujur di DATA UJI yang belum pernah dilihat proses pencarian ---
    kappa_test = hitung_kappa_untuk_parameter(
        hasil_terbaik["w_r"], hasil_terbaik["w_n"], hasil_terbaik["w_c"],
        hasil_terbaik["n_max"], hasil_terbaik["tau"],
        r_max_te, n_te, c_avg_te, y_te,
    )

    print("\n=== PARAMETER TERBAIK (dicari dari data training) ===")
    print(f"w_R    = {hasil_terbaik['w_r']}")
    print(f"w_N    = {hasil_terbaik['w_n']}")
    print(f"w_C    = {hasil_terbaik['w_c']}")
    print(f"N_max  = {hasil_terbaik['n_max']}")
    print(f"tau    = {hasil_terbaik['tau']:.4f}")
    print(f"\nCohen's Kappa (data training) : {hasil_terbaik['kappa_train']:.3f}")
    print(f"Cohen's Kappa (data uji, JUJUR): {kappa_test:.3f}")

    if kappa_test < hasil_terbaik["kappa_train"] - 0.2:
        print("\n⚠️  Kappa data uji jauh lebih rendah dari data training -- ini")
        print("    indikasi overfitting (parameter terlalu 'hafal' data training,")
        print("    tidak general). Pertimbangkan kumpulkan lebih banyak data.")

    print("\nSalin nilai w_R, w_N, w_C, N_max, tau di atas ke bagian")
    print("KONFIGURASI di main_app.py.")


if __name__ == "__main__":
    main()
