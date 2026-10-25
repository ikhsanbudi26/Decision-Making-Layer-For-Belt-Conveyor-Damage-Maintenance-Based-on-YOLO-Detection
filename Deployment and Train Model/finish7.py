"""
main_app.py - Sistem Pendukung Keputusan Tingkat Keparahan Kerusakan
Conveyor Belt Berbasis YOLOv11 dan Edge AI dengan Dashboard Real-Time.

ARSITEKTUR (revisi besar): proses deteksi (kamera + YOLO + EdgeTPU +
OpenCV) sekarang berjalan di PROSES TERPISAH TOTAL (bukan cuma thread)
dari Flask web server.

Alasan perubahan ini: pendekatan berbasis threading (meski sudah dicoba
dikunci dengan berbagai cara -- cv_lock, memisahkan encoding ke satu
thread, dst) tetap menyebabkan crash native (segfault/bus error) yang
berulang dan berganti-ganti jenis -- ciri khas race condition di level
library native (OpenCV/TFLite/EdgeTPU) yang tidak sepenuhnya bisa
dicegah cuma dengan lock Python dalam SATU proses yang sama.

Dengan proses terpisah, dua proses ini punya MEMORI TERPISAH TOTAL --
tidak ada kemungkinan dua kode native saling bertabrakan, karena secara
fisik tidak pernah berbagi ruang memori.

Komunikasi antar proses:
- Frame terbaru: proses deteksi menulis ke file JPEG di /dev/shm (RAM,
  bukan SD card -- supaya tidak boros umur SD card ditulis puluhan kali
  per detik). Flask membaca file itu tiap ada permintaan /video_feed.
- Event (sesi mulai/deteksi baru/sesi selesai/batch progress): dikirim
  lewat multiprocessing.Queue dari proses deteksi ke proses Flask, lalu
  di-relay ke dashboard lewat SSE oleh thread ringan di proses Flask
  (thread ini TIDAK PERNAH sentuh OpenCV, aman).
- Perintah (mulai sesi/mulai batch/stop batch/atur ROI): dikirim lewat
  multiprocessing.Queue dari proses Flask ke proses deteksi.

Struktur folder yang dibutuhkan:
    conveyor_project/
    ├── main_app.py          <- file ini
    ├── db_severity.py       <- modul database
    └── templates/
        └── dashboard.html

Run:  python3 main_app.py
Akses dashboard: http://<ip-raspberry-pi>:5000
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import multiprocessing as mp
import queue
import threading
import time

mp.set_start_method("spawn", force=True)

import cv2
import joblib
import numpy as np
from flask import Flask, Response, render_template, request
from ultralytics import YOLO

from db_severity import (
    init_db, buat_sesi_baru, simpan_deteksi,
    finalisasi_sesi, hitung_agregat_dari_deteksi,
)

# =======================================================
# KONFIGURASI -- WAJIB disesuaikan sebelum menjalankan
# =======================================================

MODEL_PATH = "/home/..../..../..../best_full_integer_quant_edgetpu.tflite"
DEVICE = "tpu:0"
IMGSZ = 320
CONF_THRESHOLD = 0.30

CAMERA_IP = "192.168.50.2" #jika ip berubah, bisa dilihat melalui aplikasi HiToolsDelivery
USERNAME = "admin"
PASSWORD = "@cam2626"
CHANNEL = "102" #chanel 102 disarankan dibanding 101, karena lebih lancar
RTSP_URL = f"rtsp://admin:%40cam2626@<IP>:554/Streaming/Channels/102"

ENABLE_ROI = True
ROI_COORDS_AWAL = [100, 100, 900, 600] #ukuran ROI default (bisa dirubah)

W_R, W_N, W_C = 0.6, 0.1, 0.3 #nilai berbobot rule based
TAU = 0.3221    #threshold
N_MAX = 10      #titik jenuh jumlah deteksi

DURASI_SESI_DETIK = 45

DB_PATH = "conveyor_monitoring.db"
LOGREG_MODEL_PATH = "severity_logreg.pkl"

# --- Metode yang jadi KEPUTUSAN UTAMA, ditampilkan sebagai badge di dashboard ---
# "rule_based" (default) atau "logreg" -- ganti ini SETELAH menjalankan
# bandingkan_metode.py dan tahu metode mana yang lebih sepakat dengan
# penilaian teknisi. Rule-based dan regresi logistik dua-duanya TETAP 
# dihitung dan disimpan ke database apapun pilihan di sini -- ini cuma
# menentukan mana yang ditonjolkan ke operator sebagai keputusan resmi.
METODE_UTAMA = "logreg"

# /dev/shm itu RAM (tmpfs), bukan SD card -- aman ditulis berkali-kali
# per detik tanpa mempercepat keausan SD card.
FRAME_FILE = "/dev/shm/conveyor_latest_frame.jpg"
FRAME_FILE_TMP = "/dev/shm/conveyor_latest_frame_tmp.jpg"


# =======================================================
# FUNGSI BANTU SEVERITY (dipakai proses deteksi)
# =======================================================

def roi_area(coords):
    x1, y1, x2, y2 = coords
    return max(0, (x2 - x1)) * max(0, (y2 - y1))


def bbox_overlap_ratio_in_roi(bbox, roi_coords):
    x1, y1, x2, y2 = bbox
    rx1, ry1, rx2, ry2 = roi_coords
    ox1, oy1 = max(x1, rx1), max(y1, ry1)
    ox2, oy2 = min(x2, rx2), min(y2, ry2)
    if ox1 >= ox2 or oy1 >= oy2:
        return 0.0
    overlap_area = (ox2 - ox1) * (oy2 - oy1)
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    return overlap_area / bbox_area


def klasifikasi_rule_based(r_max, n_norm, c_avg):
    score = W_R * r_max + W_N * n_norm + W_C * c_avg
    kategori = "Rusak Berat" if score > TAU else "Rusak Ringan"
    return score, kategori


def muat_model_logreg():
    if os.path.exists(LOGREG_MODEL_PATH):
        model = joblib.load(LOGREG_MODEL_PATH)
        print(f"[LOGREG] Model dimuat dari {LOGREG_MODEL_PATH}")
        return model
    print(f"[LOGREG] File {LOGREG_MODEL_PATH} belum ada -- lewati, pakai rule-based saja.")
    return None


def klasifikasi_logreg(model, r_max, n_norm, c_avg):
    if model is None:
        return None, None
    proba = model.predict_proba([[r_max, n_norm, c_avg]])[0][1]
    kategori = "Rusak Berat" if proba >= 0.5 else "Rusak Ringan"
    return kategori, float(proba)


# --- Parameter tracking: SESUAIKAN dengan kecepatan belt & FPS sungguhan ---
# IOU_MIN_TRACKING: seberapa besar overlap kotak (0-1) antar frame supaya
# dianggap objek YANG SAMA. Kriteria UTAMA -- lebih akurat dari jarak
# titik pusat saja karena memperhitungkan ukuran/bentuk kotak.
IOU_MIN_TRACKING = 0.2
# TOLERANSI_JARAK_DASAR: jarak (piksel) yang selalu ditoleransi meski
# jeda waktu ke frame terakhir sangat singkat -- meredam noise/jitter
# kecil dari deteksi YOLO itu sendiri.
TOLERANSI_JARAK_DASAR = 20
# KECEPATAN_TRACKING_PIKSEL_PER_DETIK: perkiraan seberapa jauh (piksel)
# suatu objek bisa berpindah tiap detik akibat gerakan belt. WAJIB
# disesuaikan dengan kecepatan belt & framing kamera sungguhan --
# terlalu kecil bikin satu kerusakan masih bisa dobel-hitung kalau ada
# lompatan frame (drain buffer); terlalu besar bikin dua kerusakan
# berbeda yang berdekatan keliru dianggap satu objek.
KECEPATAN_TRACKING_PIKSEL_PER_DETIK = 150
# WAKTU_HILANG_MAX_DETIK: berapa detik suatu objek boleh "tidak
# terlihat" (misal confidence sempat turun sesaat) sebelum track-nya
# dianggap benar-benar hilang/keluar ROI. Pakai DETIK, bukan hitungan
# frame -- karena jeda waktu antar frame yang diproses sekarang tidak
# selalu konsisten (ada mekanisme kuras buffer untuk real-time).
WAKTU_HILANG_MAX_DETIK = 1.0


class DetectionTracker:
    """Melacak objek kerusakan antar frame DALAM SATU SESI, supaya satu
    kerusakan fisik yang sama -- yang terlihat berulang kali di banyak
    frame berturut-turut selagi belt bergerak pelan melintasi ROI --
    cuma dihitung SEKALI sebagai satu deteksi, bukan dihitung ulang di
    setiap frame kemunculannya.

    Menggunakan IOU (overlap kotak) sebagai kriteria utama, dengan jarak
    titik pusat sebagai cadangan. Batas jarak yang ditoleransi dihitung
    berdasarkan WAKTU sungguhan yang berlalu sejak objek itu terakhir
    terlihat (bukan hitungan frame) -- supaya tetap akurat meski jeda
    antar frame yang diproses tidak konsisten (akibat mekanisme kuras
    buffer untuk menjaga deteksi tetap real-time). Kecocokan kelas
    TIDAK diwajibkan mutlak, karena YOLO kadang "berkedip"
    mengklasifikasikan objek yang sama sebagai kelas berbeda antar
    frame akibat confidence yang naik-turun tipis."""

    def __init__(self):
        self.track_aktif = {}  # track_id -> {"bbox":..., "class_name":..., "waktu_terakhir":...}
        self.id_berikutnya = 0

    def _pusat(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    def _iou(self, bbox1, bbox2):
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
        xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = max(0, x2_1 - x1_1) * max(0, y2_1 - y1_1)
        area2 = max(0, x2_2 - x1_2) * max(0, y2_2 - y1_2)
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def update(self, deteksi_frame_ini, waktu_sekarang):
        """deteksi_frame_ini: list of dict dengan key 'bbox' (x1,y1,x2,y2)
        dan 'class_name'. waktu_sekarang: time.time() saat frame ini
        diproses. Kembalikan cuma deteksi yang BENAR-BENAR BARU (objek
        yang belum pernah terlihat sebelumnya dalam sesi ini)."""

        # Buang track yang sudah lama tidak muncul lagi (kemungkinan besar
        # sudah keluar dari ROI atau memang sudah tidak ada)
        track_mati = [tid for tid, t in self.track_aktif.items()
                      if waktu_sekarang - t["waktu_terakhir"] > WAKTU_HILANG_MAX_DETIK]
        for tid in track_mati:
            del self.track_aktif[tid]

        deteksi_baru = []

        for d in deteksi_frame_ini:
            cx, cy = self._pusat(d["bbox"])

            # Kumpulkan SEMUA track yang secara posisi masuk akal jadi
            # kandidat (IOU cukup besar ATAU jarak titik pusat masih
            # dalam batas yang menyesuaikan jeda waktu) -- TANPA
            # mensyaratkan kelas sama, supaya class-flickering YOLO
            # tidak bikin objek yang sama dianggap baru.
            kandidat = []
            for tid, t in self.track_aktif.items():
                iou = self._iou(d["bbox"], t["bbox"])
                tcx, tcy = self._pusat(t["bbox"])
                jarak = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5

                selisih_waktu = waktu_sekarang - t["waktu_terakhir"]
                batas_jarak = (TOLERANSI_JARAK_DASAR +
                               KECEPATAN_TRACKING_PIKSEL_PER_DETIK * selisih_waktu)

                if iou >= IOU_MIN_TRACKING or jarak < batas_jarak:
                    kelas_cocok = (t["class_name"] == d["class_name"])
                    kandidat.append((tid, iou, jarak, kelas_cocok))

            track_cocok = None
            if kandidat:
                # Urutkan: utamakan kelas cocok, lalu IOU tertinggi,
                # lalu jarak terdekat.
                kandidat.sort(key=lambda k: (not k[3], -k[1], k[2]))
                track_cocok = kandidat[0][0]

            if track_cocok is not None:
                # Objek yang SAMA dengan sebelumnya -- cuma update posisi,
                # BUKAN deteksi baru, tidak dihitung ulang.
                self.track_aktif[track_cocok]["bbox"] = d["bbox"]
                self.track_aktif[track_cocok]["class_name"] = d["class_name"]
                self.track_aktif[track_cocok]["waktu_terakhir"] = waktu_sekarang
            else:
                # Objek baru yang belum pernah terlihat dalam sesi ini.
                tid = self.id_berikutnya
                self.id_berikutnya += 1
                self.track_aktif[tid] = {"bbox": d["bbox"], "class_name": d["class_name"],
                                          "waktu_terakhir": waktu_sekarang}
                deteksi_baru.append(d)

        return deteksi_baru


def simpan_frame_ke_file(frame):
    """Tulis frame ke file secara ATOMIC (tulis ke file temp dulu, baru
    rename) -- supaya proses Flask yang baca file ini tidak pernah dapat
    file setengah-tertulis (rename di Linux itu operasi atomic)."""
    ok, buffer = cv2.imencode('.jpg', frame)
    if ok:
        with open(FRAME_FILE_TMP, 'wb') as f:
            f.write(buffer.tobytes())
        os.replace(FRAME_FILE_TMP, FRAME_FILE)


def cek_suhu_cpu():
    """Baca suhu CPU (Celsius) langsung dari sysfs -- cara paling ringan
    & portabel di Raspberry Pi (tidak perlu shell out ke vcgencmd).
    Kembalikan None kalau gagal baca (misal dijalankan bukan di Pi)."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return None


# Ambang suhu (Celsius) -- Raspberry Pi 5 mulai throttle otomatis di
# sekitar 80-85°C. Beri peringatan SEBELUM itu terjadi, supaya operator
# tahu kalau performa mulai menurun bukan karena bug, tapi panas.
SUHU_PERINGATAN_C = 70.0


def frame_valid(frame):
    """Validasi dasar sebelum frame diproses lebih jauh (model.predict,
    cv2.rectangle, dst). Frame hasil decode H.264 yang gagal sebagian
    (lihat pesan 'error while decoding MB...' dari FFmpeg) kadang tetap
    dikembalikan cap.read() dengan status sukses, tapi datanya cacat --
    bisa menyebabkan crash native di tahap pemrosesan berikutnya kalau
    tidak disaring dulu di sini."""
    if frame is None:
        return False
    if frame.size == 0:
        return False
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        return False
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return False
    return True


# Catatan: fitur deteksi "belt bergerak/diam" sempat ditambahkan di sini,
# tapi untuk sementara DILEPAS lagi (dikeluarkan dari alur utama) --
# supaya tidak menambah kompleksitas/potensi bug baru selagi masalah
# stabilitas RTSP+H.264 lain masih ditangani. Kalau nanti mau dipasang
# lagi, logikanya: bandingkan dua frame berjarak waktu singkat di area
# ROI, kalau selisih piksel rata-rata sangat kecil, kemungkinan besar
# belt sedang diam.


# =======================================================
# PROSES DETEKSI -- berjalan TERPISAH TOTAL dari proses Flask
# =======================================================

def detection_process(command_queue, event_queue):
    """Dijalankan sebagai PROSES terpisah (bukan thread). Semua
    pemanggilan OpenCV, model YOLO, dan EdgeTPU terjadi DI SINI SAJA --
    proses Flask tidak pernah menyentuh library native ini sama sekali."""

    print(f"[MODEL] Memuat {MODEL_PATH} (device={DEVICE}, imgsz={IMGSZ})...")
    model = YOLO(MODEL_PATH)
    names = model.names
    print("[MODEL] Siap.")

    logreg_model = muat_model_logreg()

    print(f"[RTSP] Membuka stream: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("[RTSP] GAGAL membuka stream. Periksa IP, kredensial, dan jaringan.")
        return

    roi_coords = list(ROI_COORDS_AWAL)
    session_trigger = False
    batch_mode = False
    batch_sisa = None
    lokasi_sekarang = None  # diisi lewat dashboard, tersimpan di tiap sesi

    def cek_perintah_baru():
        nonlocal session_trigger, batch_mode, batch_sisa, roi_coords, lokasi_sekarang
        while True:
            try:
                cmd = command_queue.get_nowait()
            except queue.Empty:
                break
            tipe = cmd.get("cmd")
            if tipe == "start_session":
                session_trigger = True
                lokasi_sekarang = cmd.get("lokasi")
            elif tipe == "start_batch":
                batch_mode = True
                batch_sisa = cmd.get("jumlah_sesi")
                lokasi_sekarang = cmd.get("lokasi")
                session_trigger = True
            elif tipe == "stop_batch":
                batch_mode = False
            elif tipe == "set_roi":
                roi_coords = cmd["coords"]
                print(f"[ROI] Diperbarui: {roi_coords}")

    print("[SESI] Menunggu perintah 'Mulai Sesi' dari dashboard...")
    event_queue.put({"type": "menunggu"})

    while True:
        cek_perintah_baru()

        # === KONDISI IDLE ===
        if not session_trigger and not batch_mode:
            ok, frame = cap.read()
            if ok and frame_valid(frame):
                if ENABLE_ROI:
                    x1, y1, x2, y2 = roi_coords
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, "ROI (menunggu sesi)", (x1, max(y1 - 8, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                simpan_frame_ke_file(frame)
            time.sleep(0.03)
            continue

        # === MULAI SATU SESI BARU ===
        session_trigger = False

        # Cek suhu CPU -- kalau sudah tinggi, kemungkinan performa akan
        # menurun (throttling) selama sesi ini berjalan. Beri peringatan
        # ke dashboard, tapi TETAP LANJUTKAN sesi (bukan menolak) --
        # supaya operator tahu dan bisa putuskan sendiri mau lanjut atau
        # istirahatkan dulu perangkatnya.
        suhu = cek_suhu_cpu()
        if suhu is not None and suhu >= SUHU_PERINGATAN_C:
            print(f"[SUHU] Peringatan: CPU {suhu:.1f}°C -- performa berpotensi "
                  f"menurun (throttling). Pertimbangkan jeda untuk pendinginan.")
            event_queue.put({"type": "suhu_tinggi", "suhu": suhu})

        # Kuras dulu beberapa frame -- proses housekeeping database
        # (commit finalisasi_sesi + buat_sesi_baru) di titik transisi
        # antar sesi bisa menyebabkan sedikit backlog di stream RTSP
        # kalau tidak ada cap.read() selama itu. Baca & buang beberapa
        # frame di sini supaya decoder H.264 "mengejar" dulu sebelum
        # benar-benar mulai memproses -- mencegah desync yang
        # menyebabkan crash di transisi antar sesi mode batch.
        for _ in range(3):
            cap.read()

        session_id = f"S-{int(time.time())}"
        session_start = time.time()
        buat_sesi_baru(session_id, lokasi=lokasi_sekarang, db_path=DB_PATH)
        event_queue.put({"type": "sesi_mulai", "session_id": session_id,
                          "lokasi": lokasi_sekarang})
        print(f"[SESI] Mulai: {session_id} (lokasi={lokasi_sekarang}, "
              f"durasi {DURASI_SESI_DETIK} detik)")

        tracker = DetectionTracker()  # RESET tiap sesi baru -- tidak menyambung
                                        # dengan pelacakan objek di sesi sebelumnya
        frame_ke = 0
        gagal_baca_beruntun = 0  # untuk deteksi koneksi RTSP benar2 putus

        while time.time() - session_start < DURASI_SESI_DETIK:
            cek_perintah_baru()  # ROI tetap bisa diupdate di tengah sesi

            ok, frame = cap.read()
            if not ok or not frame_valid(frame):
                gagal_baca_beruntun += 1
                print(f"[RTSP] Frame gagal dibaca atau tidak valid ({gagal_baca_beruntun}x "
                      f"beruntun), dilewati...")

                # Kalau gagal berturut-turut terlalu banyak, kemungkinan
                # besar koneksi RTSP benar-benar putus (bukan cuma glitch
                # sesaat) -- buka ulang koneksinya dari nol, penting untuk
                # keandalan sesi pengumpulan data yang panjang/berulang.
                if gagal_baca_beruntun >= 20:
                    print("[RTSP] Terlalu banyak gagal beruntun -- membuka ulang koneksi...")
                    cap.release()
                    time.sleep(1.0)
                    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
                    gagal_baca_beruntun = 0
                    if not cap.isOpened():
                        print("[RTSP] Gagal membuka ulang koneksi, mencoba lagi nanti...")

                time.sleep(0.1)
                continue
            gagal_baca_beruntun = 0  # reset begitu berhasil baca frame normal lagi

            for _ in range(4):
                ok_next, frame_next = cap.read()
                if not ok_next or not frame_valid(frame_next):
                    break
                frame = frame_next

            t_mulai_proses = time.time()

            # --- Potong frame ke area ROI SEBELUM dikirim ke YOLO ---
            # Ini yang menghilangkan akar masalah "bounding box di luar ROI":
            # sebelumnya YOLO dijalankan di SELURUH frame, lalu deteksi yang
            # cuma SEBAGIAN masuk ROI (>=50% overlap) tetap lolos filter dan
            # digambar dengan koordinat ASLI-nya -- sehingga bagian yang
            # menyembul keluar ROI tetap tergambar. Dengan memotong frame
            # DULU, YOLO tidak pernah "melihat" apapun di luar ROI sama
            # sekali, jadi kotak yang dihasilkan PASTI seluruhnya di dalam
            # ROI -- tidak ada lagi kemungkinan menyembul keluar.
            if ENABLE_ROI:
                x1, y1, x2, y2 = roi_coords
                frame_untuk_deteksi = frame[y1:y2, x1:x2]
                offset_x, offset_y = x1, y1
                pakai_roi_area = roi_area(roi_coords)
            else:
                frame_untuk_deteksi = frame
                offset_x, offset_y = 0, 0
                pakai_roi_area = frame.shape[0] * frame.shape[1]

            if frame_untuk_deteksi.size == 0:
                print("[ROI] Area ROI kosong/tidak valid, dilewati...")
                simpan_frame_ke_file(frame)
                continue

            try:
                results = model.predict(
                    source=frame_untuk_deteksi, imgsz=IMGSZ, device=DEVICE,
                    conf=CONF_THRESHOLD, verbose=False,
                )
            except Exception as e:
                print(f"[INFERENSI] Gagal memproses frame ini, dilewati: {e}")
                simpan_frame_ke_file(frame)
                continue

            if ENABLE_ROI:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, "ROI", (x1, max(y1 - 8, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # --- Langkah 1: kumpulkan semua deteksi valid di frame ini ---
            # Koordinat hasil deteksi itu RELATIF terhadap frame yang sudah
            # dipotong -- geser balik (+offset_x/+offset_y) supaya pas kalau
            # digambar di atas frame ASLI (yang belum dipotong).
            deteksi_frame_ini = []
            for r in results:
                boxes = getattr(r, "boxes", None)
                if boxes is None:
                    continue
                for b in boxes:
                    bx1, by1, bx2, by2 = map(int, b.xyxy[0].tolist())
                    bx1, by1 = bx1 + offset_x, by1 + offset_y
                    bx2, by2 = bx2 + offset_x, by2 + offset_y
                    conf = float(b.conf[0])
                    cls_id = int(b.cls[0])
                    class_name = names.get(cls_id, str(cls_id))

                    bbox_area = (bx2 - bx1) * (by2 - by1)
                    area_ratio = bbox_area / pakai_roi_area if pakai_roi_area else 0.0

                    deteksi_frame_ini.append({
                        "bbox": (bx1, by1, bx2, by2), "class_name": class_name,
                        "confidence": conf, "bbox_area": bbox_area,
                        "roi_area": pakai_roi_area, "area_ratio": area_ratio,
                    })

            # --- Langkah 2: gambar SEMUA deteksi di frame ini (untuk live
            #     view -- operator tetap ingin lihat kotak di semua objek
            #     yang sedang terlihat, bukan cuma yang "baru") ---
            for d in deteksi_frame_ini:
                bx1, by1, bx2, by2 = d["bbox"]
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                label = f"{d['class_name']}: {d['confidence']:.2f}"
                cv2.putText(frame, label, (bx1, max(by1 - 8, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # --- Langkah 3: saring lewat tracker -- cuma yang BENAR-BENAR
            #     BARU (objek yang belum pernah terlihat dalam sesi ini)
            #     yang disimpan ke database & dihitung sebagai deteksi.
            #     Pakai waktu sungguhan (bukan hitungan frame), supaya
            #     tetap akurat meski jeda antar frame yang diproses
            #     tidak konsisten (ada mekanisme kuras buffer). ---
            deteksi_baru = tracker.update(deteksi_frame_ini, time.time())
            frame_ke += 1

            for d in deteksi_baru:
                simpan_deteksi(
                    session_id=session_id, class_name=d["class_name"],
                    bbox_area=d["bbox_area"], roi_area=d["roi_area"],
                    area_ratio=d["area_ratio"], confidence=d["confidence"],
                    db_path=DB_PATH,
                )
                event_queue.put({
                    "type": "deteksi_baru", "class_name": d["class_name"],
                    "confidence": d["confidence"], "area_ratio": d["area_ratio"],
                })

            simpan_frame_ke_file(frame)

            # Cetak waktu proses tiap ~30 frame -- supaya kelihatan apakah
            # sistem sanggup mengejar waktu nyata (idealnya di bawah ~66ms
            # untuk kamera 15fps, di bawah ~33ms untuk 30fps) atau tertinggal.
            durasi_proses = time.time() - t_mulai_proses
            if frame_ke % 30 == 0:
                print(f"[TIMING] Frame ke-{frame_ke}: {durasi_proses*1000:.0f} ms/frame")

        # === SESI SELESAI ===
        agregat = hitung_agregat_dari_deteksi(session_id, n_max=N_MAX, db_path=DB_PATH)

        if agregat["n"] == 0:
            finalisasi_sesi(session_id, n_detections=0, r_max=None,
                             n_norm=0.0, c_avg=None, score_s=None,
                             category_system="Tidak Rusak",
                             category_logreg=None, prob_berat_logreg=None,
                             db_path=DB_PATH)
            kategori_final, score_final = "Tidak Rusak", None
            kategori_logreg, prob_logreg = None, None
        else:
            score_final, kategori_final = klasifikasi_rule_based(
                agregat["r_max"], agregat["n_norm"], agregat["c_avg"]
            )
            kategori_logreg, prob_logreg = klasifikasi_logreg(
                logreg_model, agregat["r_max"], agregat["n_norm"], agregat["c_avg"]
            )
            finalisasi_sesi(session_id, n_detections=agregat["n"],
                             r_max=agregat["r_max"], n_norm=agregat["n_norm"],
                             c_avg=agregat["c_avg"], score_s=score_final,
                             category_system=kategori_final,
                             category_logreg=kategori_logreg,
                             prob_berat_logreg=prob_logreg, db_path=DB_PATH)

        # Tentukan kategori mana yang jadi "keputusan resmi" ditampilkan
        # ke dashboard, berdasarkan METODE_UTAMA -- otomatis fallback ke
        # rule-based kalau logreg belum tersedia (mis. model belum dilatih),
        # supaya sistem tidak pernah menampilkan status kosong.
        if METODE_UTAMA == "logreg" and kategori_logreg is not None:
            kategori_tampil = kategori_logreg
        else:
            kategori_tampil = kategori_final

        event_queue.put({
            "type": "sesi_selesai", "session_id": session_id,
            "kategori": kategori_tampil, "n_detections": agregat["n"],
            "r_max": agregat["r_max"], "c_avg": agregat["c_avg"],
            "score_s": score_final,
            "kategori_logreg": kategori_logreg, "prob_logreg": prob_logreg,
        })
        print(f"[SESI] Selesai: {session_id} -> rule-based={kategori_final}, "
              f"logreg={kategori_logreg} (n={agregat['n']})")

        if batch_mode:
            if batch_sisa is not None:
                batch_sisa -= 1
                if batch_sisa <= 0:
                    batch_mode = False
                    print("[BATCH] Selesai -- jumlah sesi target tercapai.")
                    event_queue.put({"type": "batch_selesai"})
                else:
                    event_queue.put({"type": "batch_progress", "sisa": batch_sisa})
            else:
                event_queue.put({"type": "batch_progress", "sisa": None})

            if batch_mode:
                session_trigger = True
                continue

        print("[SESI] Menunggu perintah berikutnya...")
        event_queue.put({"type": "menunggu"})


# =======================================================
# PROSES FLASK -- tidak pernah sentuh OpenCV/model sama sekali
# =======================================================

app = Flask(__name__)

sse_clients = []
sse_clients_lock = threading.Lock()

command_queue = mp.Queue()
event_queue = mp.Queue()


def broadcast_sse(data_dict):
    payload = json.dumps(data_dict)
    with sse_clients_lock:
        mati = []
        for q in sse_clients:
            try:
                q.put_nowait(payload)
            except queue.Full:
                mati.append(q)
        for q in mati:
            sse_clients.remove(q)


def relay_events_thread():
    """Thread ringan di proses Flask -- cuma pindahkan event dari
    event_queue (dikirim proses deteksi) ke semua client SSE yang
    terhubung. TIDAK PERNAH menyentuh OpenCV/model."""
    while True:
        event = event_queue.get()
        broadcast_sse(event)


@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/start_session', methods=['POST'])
def start_session():
    data = request.get_json(silent=True) or {}
    command_queue.put({"cmd": "start_session", "lokasi": data.get("lokasi")})
    return {"status": "ok"}, 200


@app.route('/start_batch', methods=['POST'])
def start_batch():
    data = request.get_json(silent=True) or {}
    jumlah = data.get("jumlah_sesi")
    lokasi = data.get("lokasi")
    command_queue.put({"cmd": "start_batch", "jumlah_sesi": int(jumlah) if jumlah else None,
                        "lokasi": lokasi})
    print(f"[BATCH] Mulai sampling -- target: {'tanpa batas' if not jumlah else f'{jumlah} sesi'}, "
          f"lokasi: {lokasi or '(tidak diisi)'}")
    return {"status": "ok"}, 200


@app.route('/stop_batch', methods=['POST'])
def stop_batch():
    command_queue.put({"cmd": "stop_batch"})
    print("[BATCH] Dihentikan manual oleh operator.")
    return {"status": "ok"}, 200


@app.route('/set_roi', methods=['POST'])
def set_roi():
    data = request.get_json()
    try:
        x1, y1, x2, y2 = int(data['x1']), int(data['y1']), int(data['x2']), int(data['y2'])
    except (KeyError, TypeError, ValueError):
        return {"status": "error", "message": "Koordinat ROI tidak valid"}, 400
    if x2 <= x1 or y2 <= y1:
        return {"status": "error", "message": "ROI harus punya lebar & tinggi positif"}, 400
    command_queue.put({"cmd": "set_roi", "coords": [x1, y1, x2, y2]})
    return {"status": "ok", "roi": [x1, y1, x2, y2]}, 200


def generate_mjpeg():
    """Baca file frame yang ditulis proses deteksi -- TIDAK PERNAH
    memanggil OpenCV sama sekali di proses/thread ini."""
    while True:
        try:
            with open(FRAME_FILE, 'rb') as f:
                frame_bytes = f.read()
        except FileNotFoundError:
            time.sleep(0.1)
            continue

        if not frame_bytes:
            time.sleep(0.05)
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)


@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                     mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stream')
def stream():
    def event_generator():
        client_queue = queue.Queue(maxsize=50)
        with sse_clients_lock:
            sse_clients.append(client_queue)
        try:
            while True:
                payload = client_queue.get()
                yield f"data: {payload}\n\n"
        finally:
            with sse_clients_lock:
                if client_queue in sse_clients:
                    sse_clients.remove(client_queue)

    return Response(event_generator(), mimetype='text/event-stream')


if __name__ == "__main__":
    # (mp.set_start_method("spawn") sudah dipanggil di bagian atas file,
    # sebelum command_queue/event_queue dibuat -- lihat komentar di sana)

    init_db(DB_PATH)

    p = mp.Process(target=detection_process, args=(command_queue, event_queue), daemon=True)
    p.start()

    t = threading.Thread(target=relay_events_thread, daemon=True)
    t.start()

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
