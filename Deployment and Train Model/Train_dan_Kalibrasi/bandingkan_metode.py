"""
bandingkan_metode.py - Perbandingan lengkap Rule-Based vs Regresi Logistik
terhadap penilaian teknisi (acuan), pada data uji yang SAMA.

Metrik yang ditampilkan:
  - Confusion matrix
  - Accuracy, Balanced accuracy
  - Precision, Recall, F1-score, Specificity, Support (per kelas)
  - Macro-average dan Weighted-average
  - Cohen's Kappa (tidak berbobot dan quadratic-weighted) + interval kepercayaan 95% (bootstrap)
  - Matthews Correlation Coefficient (MCC)
  - Uji McNemar (apakah selisih kesalahan kedua metode nyata atau kebetulan)

Dua skenario evaluasi:
  A. 3 KELAS  : semua sesi berlabel teknisi (Tidak Rusak / Ringan / Berat).
                Sesi N=0 dianggap diprediksi "Tidak Rusak" oleh kedua metode
                (logreg memang tidak menilai sesi N=0).
  B. BINER    : hanya sesi berlabel Ringan/Berat yang punya deteksi (N>=1),
                sama dengan kondisi pelatihan di train_logreg.py.

Kebutuhan : pip install numpy scikit-learn scipy matplotlib
Jalankan  : python3 bandingkan_metode.py                       (data uji sintetis)
            python3 bandingkan_metode.py --db conveyor_monitoring.db
            python3 bandingkan_metode.py --db uji.db --csv hasil.csv --grafik hasil.png
            python3 bandingkan_metode.py --db uji.db --sejak "2026-10-20 00:00:00"
              (--sejak: hanya sesi dengan timestamp >= nilai ini, untuk memisahkan data uji dari data latih)
"""
import argparse
import csv
import sqlite3
import sys

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, matthews_corrcoef,
                             precision_recall_fscore_support)

TIDAK, RINGAN, BERAT = "Tidak Rusak", "Rusak Ringan", "Rusak Berat"
KELAS_3 = [TIDAK, RINGAN, BERAT]
KELAS_2 = [RINGAN, BERAT]
N_BOOTSTRAP = 2000
SEED_BOOTSTRAP = 42


# ----------------------------------------------------------------- data
def muat_data(path_db, sejak=None, sampai=None):
    conn = sqlite3.connect(path_db)
    sql = ("SELECT session_id, timestamp, n_detections, category_system, category_teknisi, "
           "category_logreg FROM sessions WHERE category_teknisi IN (?,?,?)")
    par = list(KELAS_3)
    if sejak:
        sql += " AND timestamp >= ?"; par.append(sejak)
    if sampai:
        sql += " AND timestamp <= ?"; par.append(sampai)
    rows = conn.execute(sql + " ORDER BY timestamp", par).fetchall()
    conn.close()
    data = []
    for sid, ts, n, sys_cat, tek, lr in rows:
        if lr is None and n == 0:        # logreg tidak menilai N=0 -> setara "Tidak Rusak"
            lr = TIDAK
        data.append(dict(sid=sid, ts=ts, n=n, tek=tek, rule=sys_cat, logreg=lr))
    return data


# ----------------------------------------------------------------- metrik
def spesifisitas(cm):
    """Per kelas: TN / (TN + FP), dari confusion matrix (baris=aktual, kolom=prediksi)."""
    hasil = []
    total = cm.sum()
    for i in range(cm.shape[0]):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        hasil.append(tn / (tn + fp) if (tn + fp) else float("nan"))
    return np.array(hasil)


def kappa_ci(y, p, bobot=None, n_boot=N_BOOTSTRAP, seed=SEED_BOOTSTRAP):
    """Interval kepercayaan 95% Kappa dengan bootstrap (resampling sesi)."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    nilai = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2 and len(set(p[idx])) < 2:
            continue
        k = cohen_kappa_score(y[idx], p[idx], weights=bobot)
        if not np.isnan(k):
            nilai.append(k)
    if not nilai:
        return float("nan"), float("nan")
    return float(np.percentile(nilai, 2.5)), float(np.percentile(nilai, 97.5))


def mcnemar(y, p1, p2):
    """Uji McNemar (exact binomial) pada benar/salahnya dua metode di sesi yang sama."""
    from scipy.stats import binomtest
    y, p1, p2 = np.asarray(y), np.asarray(p1), np.asarray(p2)
    b = int(np.sum((p1 == y) & (p2 != y)))   # rule benar, logreg salah
    c = int(np.sum((p1 != y) & (p2 == y)))   # rule salah, logreg benar
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(binomtest(min(b, c), b + c, 0.5).pvalue)


def hitung(y, p, kelas):
    cm = confusion_matrix(y, p, labels=kelas)
    P, R, F, S = precision_recall_fscore_support(y, p, labels=kelas, zero_division=0)
    spes = spesifisitas(cm)
    lo, hi = kappa_ci(y, p)
    return dict(
        cm=cm, P=P, R=R, F=F, S=S, spes=spes,
        acc=accuracy_score(y, p),
        bacc=balanced_accuracy_score(y, p),
        kappa=cohen_kappa_score(y, p),
        kappa_w=cohen_kappa_score(y, p, weights="quadratic",
                                  labels=kelas) if len(kelas) > 2 else cohen_kappa_score(y, p),
        k_lo=lo, k_hi=hi,
        mcc=matthews_corrcoef(y, p),
        macro=(P.mean(), R.mean(), F.mean()),
        weighted=tuple(np.average(v, weights=S) if S.sum() else float("nan") for v in (P, R, F)),
    )


def tafsir_kappa(k):
    if np.isnan(k): return "-"
    if k < 0:    return "buruk (di bawah kebetulan)"
    if k <= 0.20: return "sangat rendah"
    if k <= 0.40: return "rendah"
    if k <= 0.60: return "sedang"
    if k <= 0.80: return "substansial"
    return "hampir sempurna"


# ----------------------------------------------------------------- cetak
def cetak_cm(nama, cm, kelas):
    lebar = max(len(k) for k in kelas) + 2
    print(f"  Confusion matrix {nama} (baris = aktual teknisi, kolom = prediksi sistem)")
    print("  " + " " * lebar + "".join(f"{k:>{lebar}}" for k in kelas))
    for i, k in enumerate(kelas):
        print("  " + f"{k:<{lebar}}" + "".join(f"{cm[i, j]:>{lebar}}" for j in range(len(kelas))))


def cetak_metode(nama, m, kelas):
    print(f"\n  --- {nama} ---")
    cetak_cm(nama, m["cm"], kelas)
    print(f"\n  {'Kelas':<14}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Spesifisitas':>14}{'Support':>9}")
    for i, k in enumerate(kelas):
        print(f"  {k:<14}{m['P'][i]:>10.3f}{m['R'][i]:>10.3f}{m['F'][i]:>10.3f}"
              f"{m['spes'][i]:>14.3f}{int(m['S'][i]):>9}")
    print(f"  {'Macro avg':<14}{m['macro'][0]:>10.3f}{m['macro'][1]:>10.3f}{m['macro'][2]:>10.3f}")
    print(f"  {'Weighted avg':<14}{m['weighted'][0]:>10.3f}{m['weighted'][1]:>10.3f}{m['weighted'][2]:>10.3f}")
    print(f"\n  Accuracy          : {m['acc']:.3f}")
    print(f"  Balanced accuracy : {m['bacc']:.3f}")
    print(f"  Cohen's Kappa     : {m['kappa']:.3f}  (CI95% {m['k_lo']:.3f} - {m['k_hi']:.3f}) -> {tafsir_kappa(m['kappa'])}")
    if len(kelas) > 2:
        print(f"  Kappa quadratic   : {m['kappa_w']:.3f}")
    print(f"  MCC               : {m['mcc']:.3f}")


def cetak_ringkas(m_rule, m_lr):
    print("\n  RINGKASAN (Rule-Based vs Regresi Logistik)")
    print(f"  {'Metrik':<20}{'Rule-Based':>12}{'Logreg':>10}{'Selisih':>10}")
    baris = [("Accuracy", m_rule["acc"], m_lr["acc"]),
             ("Balanced accuracy", m_rule["bacc"], m_lr["bacc"]),
             ("Precision (macro)", m_rule["macro"][0], m_lr["macro"][0]),
             ("Recall (macro)", m_rule["macro"][1], m_lr["macro"][1]),
             ("F1 (macro)", m_rule["macro"][2], m_lr["macro"][2]),
             ("Cohen's Kappa", m_rule["kappa"], m_lr["kappa"]),
             ("MCC", m_rule["mcc"], m_lr["mcc"])]
    for n, a, b in baris:
        print(f"  {n:<20}{a:>12.3f}{b:>10.3f}{a - b:>+10.3f}")


def evaluasi(judul, data, kelas, hasil_csv, grafik):
    print("\n" + "=" * 78)
    print(judul)
    print("=" * 78)
    if len(data) < 2:
        print("  Data tidak cukup.")
        return None
    y = [d["tek"] for d in data]
    p_rule = [d["rule"] for d in data]
    p_lr = [d["logreg"] for d in data]
    print(f"  Jumlah sesi dievaluasi: {len(data)}   "
          f"(distribusi label teknisi: " +
          ", ".join(f"{k}={y.count(k)}" for k in kelas) + ")")

    m_rule = hitung(y, p_rule, kelas)
    m_lr = hitung(y, p_lr, kelas)
    cetak_metode("Rule-Based", m_rule, kelas)
    cetak_metode("Regresi Logistik", m_lr, kelas)
    cetak_ringkas(m_rule, m_lr)

    b, c, pval = mcnemar(y, p_rule, p_lr)
    print(f"\n  Uji McNemar: rule benar & logreg salah = {b}, rule salah & logreg benar = {c}, p = {pval:.4f} "
          f"-> {'selisih signifikan (p<0,05)' if pval < 0.05 else 'selisih belum signifikan (bisa kebetulan)'}")

    kata = "tertinggi" if m_rule["kappa"] > m_lr["kappa"] else ("tertinggi" if m_lr["kappa"] > m_rule["kappa"] else "sama")
    if m_rule["kappa"] == m_lr["kappa"]:
        print("  Kesimpulan: Kappa kedua metode SAMA.")
    else:
        unggul = "Rule-Based" if m_rule["kappa"] > m_lr["kappa"] else "Regresi Logistik"
        print(f"  Kesimpulan: {unggul} memiliki Kappa {kata} ({max(m_rule['kappa'], m_lr['kappa']):.3f}).")

    for nama, m in (("Rule-Based", m_rule), ("Regresi Logistik", m_lr)):
        for i, k in enumerate(kelas):
            hasil_csv.append([judul.split(":")[0], nama, k, round(m["P"][i], 4), round(m["R"][i], 4),
                              round(m["F"][i], 4), round(m["spes"][i], 4), int(m["S"][i])])
        hasil_csv.append([judul.split(":")[0], nama, "GLOBAL", f"acc={m['acc']:.4f}",
                          f"bacc={m['bacc']:.4f}", f"kappa={m['kappa']:.4f}", f"mcc={m['mcc']:.4f}",
                          len(data)])
    grafik.append((judul.split(":")[0], kelas, m_rule, m_lr))
    return m_rule, m_lr


def buat_grafik(path, grafik):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(grafik), figsize=(6.5 * len(grafik), 4.6), squeeze=False)
    for ax, (nama, kelas, mr, ml) in zip(axes[0], grafik):
        label = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1\n(macro)", "Kappa"]
        v_r = [mr["acc"], mr["macro"][0], mr["macro"][1], mr["macro"][2], mr["kappa"]]
        v_l = [ml["acc"], ml["macro"][0], ml["macro"][1], ml["macro"][2], ml["kappa"]]
        x = np.arange(len(label)); w = 0.38
        b1 = ax.bar(x - w / 2, v_r, w, label="Rule-Based", color="#2f6fb0")
        b2 = ax.bar(x + w / 2, v_l, w, label="Regresi Logistik", color="#e08a2c")
        for bar in list(b1) + list(b2):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(label, fontsize=9)
        ax.set_ylim(0, 1.1); ax.set_title(f"Skenario {nama}"); ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Perbandingan Rule-Based vs Regresi Logistik")
    ap.add_argument("--db", default="conveyor_monitoring.db", help="berkas database data uji")
    ap.add_argument("--sejak", default=None, help="hanya sesi dengan timestamp >= nilai ini")
    ap.add_argument("--sampai", default=None, help="hanya sesi dengan timestamp <= nilai ini")
    ap.add_argument("--csv", default=None, help="simpan ringkasan ke berkas CSV")
    ap.add_argument("--grafik", default=None, help="simpan grafik perbandingan ke berkas PNG")
    a = ap.parse_args()

    try:
        data = muat_data(a.db, a.sejak, a.sampai)
    except sqlite3.Error as e:
        sys.exit(f"Gagal membuka database '{a.db}': {e}")

    print(f"Database: {a.db}   |   sesi berlabel teknisi: {len(data)}")
    tanpa_logreg = [d for d in data if d["logreg"] is None]
    if tanpa_logreg:
        print(f"Peringatan: {len(tanpa_logreg)} sesi tidak punya category_logreg dan tidak ikut dibandingkan "
              f"(bukan data uji, atau model belum dimuat saat sesi direkam).")
    data = [d for d in data if d["logreg"] is not None]
    if len(data) < 15:
        print("Data uji terlalu sedikit untuk perbandingan yang bermakna (minimal ~30 sesi).")
        if not data:
            return

    hasil_csv, grafik = [], []
    evaluasi("SKENARIO A: 3 kelas, semua sesi berlabel", data, KELAS_3, hasil_csv, grafik)
    biner = [d for d in data if d["tek"] in KELAS_2 and d["n"] >= 1]
    # pada skenario biner prediksi "Tidak Rusak" tidak mungkin (N>=1), tetapi tetap dipetakan aman
    for d in biner:
        if d["rule"] == TIDAK: d["rule"] = RINGAN
        if d["logreg"] == TIDAK: d["logreg"] = RINGAN
    evaluasi("SKENARIO B: biner Ringan vs Berat, sesi dengan deteksi (N>=1)", biner, KELAS_2, hasil_csv, grafik)

    print("\nCatatan: parameter (bobot, tau, N_max, model logreg) harus sudah final SEBELUM data uji "
          "direkam; jangan diubah setelah melihat angka di atas.")

    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["skenario", "metode", "kelas", "precision", "recall", "f1", "spesifisitas", "support"])
            w.writerows(hasil_csv)
        print(f"CSV disimpan: {a.csv}")
    if a.grafik and grafik:
        buat_grafik(a.grafik, grafik)
        print(f"Grafik disimpan: {a.grafik}")


if __name__ == "__main__":
    main()
