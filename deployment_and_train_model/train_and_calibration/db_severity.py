"""
Modul database SQLite untuk Sistem Pendukung Keputusan Tingkat Keparahan
Kerusakan Conveyor Belt.

Dua tabel:
- sessions   : satu baris per sesi inspeksi (ringkasan + kategori keputusan)
- detections : satu baris per objek kerusakan terdeteksi dalam suatu sesi

Catatan penting soal NULL:
Untuk sesi dengan n_detections=0 ("Tidak Rusak"), kolom r_max, n_norm,
c_avg, dan score_s sengaja diisi None (jadi NULL di SQLite) -- BUKAN 0 --
karena R_max/C_avg/S memang tidak terdefinisi secara matematis kalau
tidak ada deteksi sama sekali. Ini penting supaya fungsi agregat SQL
(AVG, MAX) nanti tidak bias oleh angka 0 yang keliru.
"""

import sqlite3
from datetime import datetime

DB_PATH = "conveyor_monitoring.db"


def get_connection(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")  # SQLite tidak enforce FK secara default
    # WAL (Write-Ahead Logging) jauh lebih cepat untuk pola tulis kita
    # (banyak commit kecil berturut-turut) dibanding mode default --
    # penting supaya jeda commit tidak sempat bikin RTSP desync di
    # thread pemanggil (lihat catatan di main_app.py soal transisi sesi).
    conn.execute("PRAGMA journal_mode = WAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=DB_PATH):
    conn = get_connection(db_path)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            timestamp DATETIME NOT NULL,
            lokasi TEXT,
            n_detections INTEGER NOT NULL,
            r_max REAL,
            n_norm REAL,
            c_avg REAL,
            score_s REAL,
            category_system TEXT NOT NULL,
            category_teknisi TEXT,
            category_logreg TEXT,
            prob_berat_logreg REAL,
            catatan TEXT
        )
    """)

    # Migrasi utk database yang sudah dibuat sebelum kolom ini ditambahkan --
    # CREATE TABLE IF NOT EXISTS tidak menambah kolom baru ke tabel lama,
    # jadi perlu ALTER TABLE terpisah. Dibungkus try/except karena SQLite
    # error kalau kolom itu sudah ada (tidak ada sintaks "ADD COLUMN IF NOT EXISTS").
    for kolom, tipe in [("category_logreg", "TEXT"), ("prob_berat_logreg", "REAL"),
                         ("lokasi", "TEXT")]:
        try:
            c.execute(f"ALTER TABLE sessions ADD COLUMN {kolom} {tipe}")
        except sqlite3.OperationalError:
            pass  # kolom sudah ada, aman diabaikan

    c.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            class_name TEXT NOT NULL,
            bbox_area REAL NOT NULL,
            roi_area REAL NOT NULL,
            area_ratio REAL NOT NULL,
            confidence REAL NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """)

    # Index mempercepat query "ambil semua deteksi milik satu sesi"
    c.execute("CREATE INDEX IF NOT EXISTS idx_detections_session ON detections(session_id)")

    conn.commit()
    conn.close()
    print(f"Database siap di: {db_path}")


def buat_sesi_baru(session_id, lokasi=None, db_path=DB_PATH):
    """Panggil ini di AWAL sesi, sebelum ada deteksi apapun -- supaya
    baris sessions sudah ada duluan (memenuhi FOREIGN KEY di tabel
    detections). Kolom hasil severity diisi placeholder dulu, nanti
    dilengkapi lewat finalisasi_sesi().

    lokasi: opsional, nama/ID unit conveyor yang sedang diamati (mis.
    "Belt A", "Belt B") -- berguna kalau kamu sampling dari beberapa
    unit conveyor berbeda, supaya bisa dibedakan saat analisis nanti."""
    conn = get_connection(db_path)
    conn.execute("""
        INSERT INTO sessions
        (session_id, timestamp, lokasi, n_detections, r_max, n_norm, c_avg, score_s, category_system)
        VALUES (?, ?, ?, 0, NULL, NULL, NULL, NULL, 'Diproses')
    """, (session_id, datetime.now().isoformat(), lokasi))
    conn.commit()
    conn.close()


def finalisasi_sesi(session_id, n_detections, r_max, n_norm, c_avg, score_s,
                     category_system, category_logreg=None, prob_berat_logreg=None,
                     db_path=DB_PATH):
    """Panggil ini di AKHIR sesi, setelah semua deteksi terkumpul dan
    severity sudah dihitung -- melengkapi baris yang dibuat buat_sesi_baru().

    category_logreg & prob_berat_logreg bersifat OPSIONAL -- diisi None
    kalau model regresi logistik belum ada/belum dilatih (wajar di awal,
    sebelum data kalibrasi teknisi cukup terkumpul)."""
    conn = get_connection(db_path)
    conn.execute("""
        UPDATE sessions
        SET n_detections = ?, r_max = ?, n_norm = ?, c_avg = ?,
            score_s = ?, category_system = ?, category_logreg = ?,
            prob_berat_logreg = ?
        WHERE session_id = ?
    """, (n_detections, r_max, n_norm, c_avg, score_s, category_system,
          category_logreg, prob_berat_logreg, session_id))
    conn.commit()
    conn.close()


def simpan_deteksi(session_id, class_name, bbox_area, roi_area, area_ratio,
                    confidence, db_path=DB_PATH):
    """Simpan satu baris deteksi mentah, dipanggil sekali per objek
    kerusakan yang muncul dalam sesi. Timestamp presisi tinggi disimpan
    supaya urutan kemunculan deteksi DALAM satu sesi bisa direkonstruksi
    untuk visualisasi akumulasi/tren singkat di dashboard."""
    conn = get_connection(db_path)
    conn.execute("""
        INSERT INTO detections
        (session_id, timestamp, class_name, bbox_area, roi_area, area_ratio, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (session_id, datetime.now().isoformat(), class_name, bbox_area,
          roi_area, area_ratio, confidence))
    conn.commit()
    conn.close()


def ambil_tren_dalam_sesi(session_id, db_path=DB_PATH):
    """Ambil seluruh deteksi milik satu sesi, terurut waktu -- untuk
    ditampilkan sebagai grafik akumulasi selama sesi berlangsung
    (mis. jumlah deteksi kumulatif, atau R_max yang ter-update tiap
    ada deteksi baru). BUKAN untuk tren lintas sesi/hari."""
    conn = get_connection(db_path)
    rows = conn.execute("""
        SELECT timestamp, class_name, area_ratio, confidence
        FROM detections
        WHERE session_id = ?
        ORDER BY timestamp
    """, (session_id,)).fetchall()
    conn.close()

    hasil = []
    r_max_berjalan = 0.0
    for i, row in enumerate(rows, start=1):
        r_max_berjalan = max(r_max_berjalan, row["area_ratio"])
        hasil.append({
            "urutan_ke": i,
            "timestamp": row["timestamp"],
            "class_name": row["class_name"],
            "area_ratio": row["area_ratio"],
            "confidence": row["confidence"],
            "r_max_kumulatif": r_max_berjalan,
        })
    return hasil


def hitung_agregat_dari_deteksi(session_id, n_max=5, db_path=DB_PATH):
    """Hitung R_max, N_norm, C_avg langsung lewat query agregat SQL,
    dari baris-baris di tabel detections milik satu sesi -- daripada
    dihitung manual di kode Python."""
    conn = get_connection(db_path)
    row = conn.execute("""
        SELECT
            COUNT(*) AS n,
            MAX(area_ratio) AS r_max,
            AVG(confidence) AS c_avg
        FROM detections
        WHERE session_id = ?
    """, (session_id,)).fetchone()
    conn.close()

    if row["n"] == 0:
        return {"n": 0, "r_max": None, "n_norm": 0.0, "c_avg": None}

    n_norm = min(row["n"] / n_max, 1.0)
    return {"n": row["n"], "r_max": row["r_max"], "n_norm": n_norm, "c_avg": row["c_avg"]}


def ambil_sesi_terbaru(limit=1, db_path=DB_PATH):
    """Untuk ditampilkan di dashboard -- cukup sesi paling akhir,
    bukan riwayat panjang."""
    conn = get_connection(db_path)
    rows = conn.execute("""
        SELECT * FROM sessions ORDER BY timestamp DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def ambil_data_kalibrasi(db_path=DB_PATH):
    """Ambil seluruh sesi yang SUDAH diberi label teknisi -- ini yang
    dipakai untuk kalibrasi bobot rule-based, training regresi logistik,
    dan perhitungan Cohen's Kappa. Sesi tanpa label teknisi (belum
    dinilai) otomatis tidak ikut."""
    conn = get_connection(db_path)
    rows = conn.execute("""
        SELECT session_id, n_detections, r_max, n_norm, c_avg, score_s,
               category_system, category_teknisi, catatan
        FROM sessions
        WHERE category_teknisi IS NOT NULL
        ORDER BY timestamp
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_label_teknisi(session_id, category_teknisi, catatan=None, db_path=DB_PATH):
    """Dipanggil setelah teknisi menilai suatu sesi secara blind --
    mengisi kolom category_teknisi yang tadinya NULL."""
    conn = get_connection(db_path)
    conn.execute("""
        UPDATE sessions SET category_teknisi = ?, catatan = ?
        WHERE session_id = ?
    """, (category_teknisi, catatan, session_id))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()

    # --- Contoh pemakaian ---
    buat_sesi_baru("S001")
    finalisasi_sesi("S001", n_detections=0, r_max=None, n_norm=0.0,
                     c_avg=None, score_s=None, category_system="Tidak Rusak")

    buat_sesi_baru("S002")
    simpan_deteksi("S002", "Tear", bbox_area=52800, roi_area=240000,
                    area_ratio=0.22, confidence=0.81)
    agregat = hitung_agregat_dari_deteksi("S002")
    print("Agregat sesi S002:", agregat)

    finalisasi_sesi("S002", n_detections=agregat["n"], r_max=agregat["r_max"],
                     n_norm=agregat["n_norm"], c_avg=agregat["c_avg"],
                     score_s=0.375, category_system="Rusak Berat")

    print("\nSesi terbaru:", ambil_sesi_terbaru(limit=5))
