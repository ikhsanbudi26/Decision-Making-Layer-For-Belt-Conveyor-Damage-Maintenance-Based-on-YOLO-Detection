"""
Database sintetis untuk eksperimen/pembanding pipeline (train_logreg.py,
kalibrasi, bandingkan_metode.py). Bukan data lapangan.

Skenario: Unit 1 = Rusak Ringan, Unit 2 = Rusak Berat, Unit 3 = Tidak Rusak
(label teknisi per unit).

DUA MODE:
  --mode latih (bawaan) : data pelatihan/kalibrasi. category_logreg dan
                          prob_berat_logreg dikosongkan (NULL), seperti sesi
                          yang direkam sebelum model regresi logistik dimuat.
  --mode uji            : data pengujian (data baru setelah model dimuat).
                          category_logreg dan prob_berat_logreg TERISI otomatis
                          oleh model regresi logistik, sehingga siap dipakai
                          menghitung Cohen's Kappa kedua metode.

Sumber model regresi logistik pada mode uji:
  - bawaan : dilatih di sini dari data latih sintetis (generator yang sama,
             pemisahan 70:30 berstrata seperti train_logreg.py), atau
  - --model severity_logreg.pkl : pakai model berkas yang sudah ada.

DURASI SESI: atur lewat DURASI_SESI_DETIK di bawah atau --durasi (detik).
Sesi berurutan dalam satu unit dimulai tepat setiap durasi sesi (+ jeda tetap
JEDA_ANTAR_SESI_DETIK / --jeda, bawaan 0), dan waktu deteksi tersebar di dalam
sesi. Sebaran jumlah deteksi, R_max, dan confidence TIDAK ikut berubah
mengikuti durasi (distribusinya tetap).

Contoh:
  python3 generate_synthetic_db.py                       # data latih
  python3 generate_synthetic_db.py --mode uji            # data uji, logreg terisi
  python3 generate_synthetic_db.py --mode uji --durasi 15 --seed 7
"""
import argparse, random, sqlite3, os, sys
from datetime import datetime, timedelta

SEED = 1316                     # seed data latih
SEED_UJI = 147                  # seed data uji (beda dari data latih)
OUT_LATIH = "conveyor_monitoring_SYNTHETIC.db"
OUT_UJI = "conveyor_monitoring_SYNTHETIC_UJI.db"
ROI_AREA = 128870.0
W_R, W_N, W_C, TAU, N_MAX = 0.6, 0.1, 0.3, 0.3221, 10   # parameter main_app.py
CLASSES = ["Tear", "Hole", "Puncture", "Patch Work"]

DURASI_SESI_DETIK = 45          # durasi satu sesi (detik); samakan dengan main_app.py
JEDA_ANTAR_SESI_DETIK = 0       # jeda tetap antar sesi (detik). 0 = sesi berurutan tepat
                                # setiap DURASI_SESI_DETIK; isi 1-2 untuk meniru jeda transisi

# (unit, tanggal/jam mulai, label teknisi, jumlah sesi)
PLAN = [
    ("Unit 1", datetime(2026, 9, 23, 14, 9),  "ringan", 65),
    ("Unit 2", datetime(2026, 9, 23, 15, 55), "berat", 50),
    ("Unit 3", datetime(2026, 10, 5, 15, 30),  "tidak", 15),
]
# Data uji: direkam SESUDAH model dimuat (tanggal setelah data latih)
PLAN_UJI = [
    ("Unit 1", datetime(2026, 10, 6, 15, 15),  "ringan", 45),
    ("Unit 2", datetime(2026, 10, 7, 13, 20), "berat", 45),
    ("Unit 3", datetime(2026, 10, 7, 15, 50),   "tidak", 15),
]
LABEL = {"tidak": "Tidak Rusak", "ringan": "Rusak Ringan", "berat": "Rusak Berat"}
NOTES = {"Rusak Berat": ["bagian tepi aus", "retak memanjang", None, None, None],
         "Rusak Ringan": ["goresan permukaan", None, None, None, None, None],
         "Tidak Rusak": [None] * 8}

def clip(x, lo, hi): return max(lo, min(hi, x))

def make_detections(rng, kondisi):
    """list (kelas, area_ratio, conf); sengaja ada tumpang tindih antar kategori."""
    dets = []
    if kondisi == "tidak":
        if rng.random() < 0.12:      # false positive / tambalan lama
            dets.append((rng.choice(["Patch Work", "Puncture", "Tear"]),
                         clip(rng.gauss(0.02, 0.01), 0.004, 0.06), clip(rng.gauss(0.42, 0.08), 0.30, 0.65)))
        return dets
    if kondisi == "ringan":
        if rng.random() < 0.08:      # kerusakan tidak terdeteksi model
            return dets
        n = rng.choices([1, 2, 3], weights=[66, 28, 6])[0]
        for _ in range(n):
            dets.append((rng.choices(CLASSES, weights=[30, 25, 30, 15])[0],
                         clip(rng.lognormvariate(-3.1, 0.6), 0.005, 0.22),
                         clip(rng.gauss(0.60, 0.13), 0.30, 0.92)))
        if rng.random() < 0.14:      # terbaca model lebih parah dari penilaian teknisi
            dets[0] = (dets[0][0], clip(rng.gauss(0.42, 0.06), 0.28, 0.58), clip(rng.gauss(0.85, 0.05), 0.7, 0.95))
            if len(dets) < 2:
                dets.append((rng.choice(CLASSES), clip(rng.gauss(0.10, 0.03), 0.04, 0.2), clip(rng.gauss(0.75, 0.08), 0.5, 0.95)))
        return dets
    if rng.random() < 0.05:          # kerusakan berat tidak terdeteksi
        return dets
    n = rng.choices([1, 2, 3, 4, 5], weights=[16, 38, 30, 12, 4])[0]
    for _ in range(n):
        dets.append((rng.choices(CLASSES, weights=[40, 30, 20, 10])[0],
                     clip(rng.lognormvariate(-1.3, 0.6), 0.03, 0.55),
                     clip(rng.gauss(0.76, 0.11), 0.34, 0.96)))
    return dets

def agregat(dets):
    n = len(dets)
    if n == 0:
        return 0, 0.0, 0.0, 0.0, 0.0   # sesi tanpa deteksi: semua nilai 0 (bukan NULL)
    r_max = max(d[1] for d in dets)
    n_norm = min(n / N_MAX, 1.0)
    c_avg = sum(d[2] for d in dets) / n
    return n, r_max, n_norm, c_avg, W_R * r_max + W_N * n_norm + W_C * c_avg

def build(seed, plan=None, durasi=None, idx_awal=0, jeda=None):
    plan = PLAN if plan is None else plan
    durasi = DURASI_SESI_DETIK if durasi is None else durasi
    jeda = JEDA_ANTAR_SESI_DETIK if jeda is None else jeda
    rng = random.Random(seed)
    sesi, idx = [], idx_awal
    for unit, start, kond, jumlah in plan:
        t = start
        for _ in range(jumlah):
            idx += 1
            dets = make_detections(rng, kond)
            n, r_max, n_norm, c_avg, s = agregat(dets)
            cat = "Tidak Rusak" if n == 0 else ("Rusak Berat" if s > TAU else "Rusak Ringan")
            ts_det = []
            for k in range(len(dets)):
                # waktu deteksi di dalam sesi, dijaga tidak melewati durasi sesi
                offset = min(2 + k * rng.randint(2, 4), max(durasi - 1, 1))
                ts_det.append(t + timedelta(seconds=offset))
            sesi.append(dict(sid=f"S-{idx:04d}", t=t, unit=unit, dets=sorted(dets, key=lambda d: d[1]),
                             ts_det=ts_det, n=n, r_max=r_max, n_norm=n_norm, c_avg=c_avg, s=s,
                             cat_sys=cat, cat_tek=LABEL[kond], note=rng.choice(NOTES[LABEL[kond]])))
            rng.randint(9, 22)   # dibuang: hanya menjaga urutan angka acak (jadi isi data
                                 # per seed) tetap sama dengan versi lama yang jedanya acak
            t += timedelta(seconds=durasi + jeda)   # awal sesi berikutnya: kelipatan durasi sesi
    return sesi

def prediksi_logreg(model, s):
    """Prediksi regresi logistik untuk satu sesi. Model hanya dilatih untuk
    membedakan Ringan vs Berat, jadi sesi tanpa deteksi (N = 0) TIDAK dijalankan
    lewat model: langsung ditulis "Tidak Rusak" dengan peluang Berat 0.0
    (sama seperti penyaring N = 0 pada rule-based). Tanpa model (mode latih)
    kedua kolom dibiarkan NULL."""
    if model is None:
        return None, None
    if s["n"] == 0:
        return "Tidak Rusak", 0.0
    proba = float(model.predict_proba([[s["r_max"], s["n_norm"], s["c_avg"]]])[0][1])
    return ("Rusak Berat" if proba >= 0.5 else "Rusak Ringan"), round(proba, 4)


def write_db(sesi, out, model=None):
    if os.path.exists(out): os.remove(out)
    conn = sqlite3.connect(out); c = conn.cursor()
    c.executescript("""
    CREATE TABLE sessions (
        session_id TEXT PRIMARY KEY, timestamp DATETIME NOT NULL, lokasi TEXT,
        n_detections INTEGER NOT NULL, r_max REAL, n_norm REAL, c_avg REAL, score_s REAL,
        category_system TEXT NOT NULL, category_teknisi TEXT,
        category_logreg TEXT, prob_berat_logreg REAL, catatan TEXT
    );
    CREATE TABLE detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, class_name TEXT,
        bbox_area REAL, roi_area REAL, area_ratio REAL, confidence REAL, timestamp DATETIME
    );""")
    for s in sesi:
        for (cls, ar, cf), ts in zip(s["dets"], s["ts_det"]):
            c.execute("INSERT INTO detections (session_id,class_name,bbox_area,roi_area,area_ratio,confidence,timestamp) VALUES (?,?,?,?,?,?,?)",
                      (s["sid"], cls, round(ar * ROI_AREA, 1), ROI_AREA, round(ar, 6), round(cf, 4), ts.isoformat(sep=" ")))
        cat_lr, prob_lr = prediksi_logreg(model, s)
        c.execute("""INSERT INTO sessions (session_id,timestamp,lokasi,n_detections,r_max,n_norm,c_avg,score_s,
                     category_system,category_teknisi,category_logreg,prob_berat_logreg,catatan)
                     VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                  (s["sid"], s["t"].isoformat(sep=" "), s["unit"], s["n"],
                   round(s["r_max"], 6), s["n_norm"],
                   round(s["c_avg"], 4),
                   round(s["s"], 6), s["cat_sys"], s["cat_tek"], cat_lr, prob_lr, s["note"]))
    conn.commit(); conn.close()

def latih_logreg_dari_sintetis(seed_latih, durasi):
    """Latih regresi logistik seperti train_logreg.py: sesi Ringan/Berat yang
    punya deteksi, pemisahan 70:30 berstrata (random_state=42), fit di 70%."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    data = [x for x in build(seed_latih, PLAN, durasi)
            if x["cat_tek"] in ("Rusak Ringan", "Rusak Berat") and x["n"] >= 1]
    X = np.array([[x["r_max"], x["n_norm"], x["c_avg"]] for x in data])
    y = np.array([1 if x["cat_tek"] == "Rusak Berat" else 0 for x in data])
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    model = LogisticRegression()
    model.fit(X_tr, y_tr)
    return model, len(X_tr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generator database sintetis (latih / uji)")
    ap.add_argument("--mode", choices=["latih", "uji"], default="latih")
    ap.add_argument("--seed", type=int, default=None, help="seed data (bawaan: sesuai mode)")
    ap.add_argument("--durasi", type=int, default=DURASI_SESI_DETIK, help="durasi satu sesi (detik)")
    ap.add_argument("--jeda", type=int, default=JEDA_ANTAR_SESI_DETIK, help="jeda tetap antar sesi (detik)")
    ap.add_argument("--model", default=None, help="(mode uji) berkas severity_logreg.pkl yang sudah ada")
    ap.add_argument("--out", default=None, help="nama berkas database keluaran")
    a = ap.parse_args()

    if a.mode == "latih":
        seed = SEED if a.seed is None else a.seed
        sesi = build(seed, PLAN, a.durasi, jeda=a.jeda)
        out = a.out or OUT_LATIH
        write_db(sesi, out, model=None)
        info = "category_logreg/prob_berat_logreg = NULL"
    else:
        seed = SEED_UJI if a.seed is None else a.seed
        if a.model:
            import joblib
            model, info_model = joblib.load(a.model), f"model dimuat dari {a.model}"
        else:
            model, n_tr = latih_logreg_dari_sintetis(SEED, a.durasi)
            info_model = f"model dilatih dari data latih sintetis (seed {SEED}, {n_tr} sesi latih)"
        n_latih = sum(j for _, _, _, j in PLAN)          # nomor sesi uji melanjutkan data latih
        sesi = build(seed, PLAN_UJI, a.durasi, idx_awal=n_latih, jeda=a.jeda)
        out = a.out or OUT_UJI
        write_db(sesi, out, model=model)
        info = f"category_logreg/prob_berat_logreg terisi ({info_model})"

    print(f"mode: {a.mode} | seed: {seed} | durasi sesi: {a.durasi} dtk | sesi: {len(sesi)} "
          f"| deteksi: {sum(len(x['dets']) for x in sesi)}")
    print(f"keluaran: {out}\n{info}")
