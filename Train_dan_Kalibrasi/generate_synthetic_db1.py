"""
Database sintetis untuk eksperimen/pembanding pipeline (train_logreg.py,
kalibrasi, bandingkan_metode.py). Bukan data lapangan.

Skenario: Unit 1 = Rusak Ringan, Unit 2 = Rusak Berat, Unit 3 = Tidak Rusak
(label teknisi per unit), 41 sesi per unit = 123 sesi.
Jalankan: python3 generate_synthetic_db.py [seed]
"""
import random, sqlite3, os, sys
from datetime import datetime, timedelta

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1316
OUT = "conveyor_monitoring_SYNTHETIC.db"
ROI_AREA = 128870.0
W_R, W_N, W_C, TAU, N_MAX = 0.4, 0.3, 0.3, 0.4636, 5   # parameter main_app.py
CLASSES = ["Tear", "Hole", "Puncture", "Patch Work"]

# (unit, tanggal/jam mulai, label teknisi, jumlah sesi)
PLAN = [
    ("Unit 1", datetime(2026, 9, 23, 15, 57),  "ringan", 65),
    ("Unit 2", datetime(2026, 10, 5, 15, 2), "berat", 50),
    ("Unit 3", datetime(2026, 10, 6, 15, 30),  "tidak", 15),
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
        return 0, None, 0.0, None, None
    r_max = max(d[1] for d in dets)
    n_norm = min(n / N_MAX, 1.0)
    c_avg = sum(d[2] for d in dets) / n
    return n, r_max, n_norm, c_avg, W_R * r_max + W_N * n_norm + W_C * c_avg

def build(seed):
    rng = random.Random(seed)
    sesi, idx = [], 0
    for unit, start, kond, jumlah in PLAN:
        t = start
        for _ in range(jumlah):
            idx += 1
            dets = make_detections(rng, kond)
            n, r_max, n_norm, c_avg, s = agregat(dets)
            cat = "Tidak Rusak" if n == 0 else ("Rusak Berat" if s > TAU else "Rusak Ringan")
            ts_det = []
            for k in range(len(dets)):
                ts_det.append(t + timedelta(seconds=2 + k * rng.randint(2, 4)))
            sesi.append(dict(sid=f"S-{idx:04d}", t=t, unit=unit, dets=sorted(dets, key=lambda d: d[1]),
                             ts_det=ts_det, n=n, r_max=r_max, n_norm=n_norm, c_avg=c_avg, s=s,
                             cat_sys=cat, cat_tek=LABEL[kond], note=rng.choice(NOTES[LABEL[kond]])))
            t += timedelta(seconds=15 + rng.randint(9, 22))
    return sesi

def write_db(sesi):
    if os.path.exists(OUT): os.remove(OUT)
    conn = sqlite3.connect(OUT); c = conn.cursor()
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
        c.execute("""INSERT INTO sessions (session_id,timestamp,lokasi,n_detections,r_max,n_norm,c_avg,score_s,
                     category_system,category_teknisi,category_logreg,prob_berat_logreg,catatan)
                     VALUES (?,?,?,?,?,?,?,?,?,?,NULL,NULL,?)""",
                  (s["sid"], s["t"].isoformat(sep=" "), s["unit"], s["n"],
                   None if s["r_max"] is None else round(s["r_max"], 6), s["n_norm"],
                   None if s["c_avg"] is None else round(s["c_avg"], 4),
                   None if s["s"] is None else round(s["s"], 6), s["cat_sys"], s["cat_tek"], s["note"]))
    conn.commit(); conn.close()

if __name__ == "__main__":
    sesi = build(SEED)
    write_db(sesi)
    print("seed:", SEED, "| sesi:", len(sesi), "| deteksi:", sum(len(s["dets"]) for s in sesi))
