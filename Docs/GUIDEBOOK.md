# 📋 Guidebook - Sistem Pendukung Keputusan Tingkat Keparahan Kerusakan Conveyor Belt

**Berbasis YOLOv11 dan Edge AI dengan Dashboard Pemantauan Real-Time**

Panduan lengkap instalasi, konfigurasi, dan penggunaan. Masalah yang muncul: lihat [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).

> **Dokumen ini dikembangkan dari repository wise78.** Tahap instalasi Raspberry Pi, setup IP camera RTSP, dan seluruh lapisan deteksi YOLO (pengumpulan data, pelatihan, konversi Edge TPU) mengikuti prosedur di:
> **<https://github.com/wise78/Conveyor-Belt-Damage-Detection>** (`docs/setup_guide.md` dan `docs/troubleshooting.md`).
> Prosedur itu tidak diulang di sini. Yang dijelaskan lengkap di sini adalah bagian **khas proyek ini**: lapisan keputusan, database, dashboard, kalibrasi, dan pengujian. Perbedaan penting dengan repository wise78 ada di [bagian 3](#3--perbedaan-dengan-repository-wise78).

## 📑 Daftar Isi

1. [Overview](#1--overview)
2. [Prerequisites (Hardware dan Software)](#2--prerequisites)
3. [Perbedaan dengan repository wise78](#3--perbedaan-dengan-repository-wise78)
4. [Step 1: Setup Raspberry Pi](#4--step-1-setup-raspberry-pi)
5. [Step 2: Setup IP Camera (RTSP)](#5--step-2-setup-ip-camera-rtsp)
6. [Step 3: Lapisan Deteksi YOLO](#6--step-3-lapisan-deteksi-yolo)
7. [Step 4: Konfigurasi Sistem (`main_app.py`)](#7--step-4-konfigurasi-sistem)
8. [Step 5: Lapisan Keputusan](#8--step-5-lapisan-keputusan-tingkat-keparahan)
9. [Step 6: Database](#9--step-6-database-sqlite)
10. [Step 7: Menjalankan Sistem dan Dashboard](#10--step-7-menjalankan-sistem-dan-dashboard)
11. [Step 8: Pengumpulan Data, Kalibrasi, dan Pengujian](#11--step-8-pengumpulan-data-kalibrasi-dan-pengujian)
12. [Simulator Tanpa Raspberry Pi](#12--simulator-tanpa-raspberry-pi)
13. [Configuration Details](#13--configuration-details)
14. [Maintenance](#14--maintenance)
15. [Daftar Berkas](#15--daftar-berkas)

---

## 1. 🎯 Overview

Sistem mengamati permukaan conveyor belt secara real-time, mendeteksi kerusakannya, lalu menyimpulkan **satu keputusan keparahan**: **Tidak Rusak**, **Rusak Ringan**, atau **Rusak Berat**.

### Empat lapisan

```
┌────────────┐   ┌───────────────┐   ┌───────────────────┐   ┌──────────────┐
│ 1. Kamera  │ → │ 2. Deteksi    │ → │ 3. Keputusan      │ → │ 4. Dashboard │
│ IP / RTSP  │   │ YOLOv11n +    │   │ Rule-based SAW &  │   │ Flask + SSE  │
│ Hikvision  │   │ Edge TPU      │   │ Regresi Logistik  │   │ + SQLite     │
└────────────┘   └───────────────┘   └───────────────────┘   └──────────────┘
```

| Lapisan | Fungsi | Dijelaskan di |
|---|---|---|
| 1. Kamera | Mengambil video belt | Repository wise78 + Step 2 |
| 2. Deteksi | Mendeteksi dan melacak kerusakan | Repository wise78 + Step 3 |
| 3. Keputusan | Menghitung keparahan per sesi | Step 5 (khas proyek ini) |
| 4. Dashboard | Menampilkan dan menyimpan hasil | Step 6-7 (khas proyek ini) |

### Empat kelas kerusakan

`Hole`, `Puncture`, `Tear`, `Patch Work`.

### Konsep sesi

Satu **sesi** adalah satu jendela pengamatan berdurasi tetap (`DURASI_SESI_DETIK`). Tiap sesi menghasilkan:

| Simbol | Arti |
|---|---|
| `N` | Jumlah kerusakan unik (setelah tracking) |
| `R_max` | Rasio luas bounding box terbesar terhadap luas ROI |
| `C_avg` | Rata-rata confidence deteksi |
| `S` | Skor keparahan gabungan |

Satu sesi hanya mencakup sebagian permukaan belt. Untuk keputusan satu putaran penuh gunakan **Mode Custom Sesi** (Step 7): beberapa sesi berturut-turut, keputusan akhir memakai prinsip *worst-case*.

---

## 2. 🔧 Prerequisites

### Hardware Requirements

#### Untuk pengumpulan data dan inferensi

| Komponen | Spesifikasi |
|---|---|
| Komputer edge | **Raspberry Pi 5** (4 GB minimum, 8 GB disarankan) |
| Akselerator | **Google Coral USB Accelerator** (Edge TPU) |
| Kamera | **IP Camera Hikvision** (atau kompatibel RTSP), disarankan PoE |
| Penyimpanan | MicroSD ≥ 32 GB (Class 10 / A1) |
| Catu daya | **5 V / 5 A** resmi untuk Pi 5 |
| Jaringan | Kabel Ethernet untuk kamera; Pi dan kamera satu subnet |
| Pendingin | Heatsink dan kipas |

#### Untuk pelatihan model (opsional, bukan di Pi)

- Google Colab dengan GPU (lihat repository wise78)
- Dataset berlabel kerusakan conveyor belt (proyek ini: dataset publik Roboflow)

### Software Requirements

#### Raspberry Pi

| Komponen | Versi / Keterangan |
|---|---|
| OS | Raspberry Pi OS **Bookworm** 64-bit |
| Python | **3.9.18** lewat `pyenv`, dalam venv `conveyor_env` |
| Edge TPU runtime | `libedgetpu1-std` (apt, repo Google Coral) |
| PyCoral | `pycoral 2.0.0` |
| TFLite | `tflite-runtime` |
| Pustaka | `opencv-python`, `numpy`, `flask`, `joblib`, `scikit-learn`, `scipy`, `matplotlib` |
| Database | SQLite (bawaan Python) |

#### Google Colab

Akun Google dan runtime GPU. Versi paket pelatihan mengikuti repository wise78 (`ultralytics`, `tensorflow`, `onnx`, `onnxslim`).

---

## 3. 🔀 Perbedaan dengan repository wise78

Repository wise78 menjadi rujukan, tetapi lingkungan proyek ini berbeda. **Ikuti prosedur wise78, lalu terapkan penyesuaian berikut:**

| Hal | `setup_guide.md` wise78 | Proyek ini |
|---|---|---|
| Raspberry Pi | Pi 4 | **Pi 5** |
| Raspberry Pi OS | Bullseye atau Bookworm | **Bookworm** |
| Python 3.9 (Step 1.2) | `apt install python3.9 python3.9-venv ...` | Python 3.9 tidak tersedia lewat apt di Bookworm (bawaannya 3.11) → pasang lewat **`pyenv`** |
| Edge TPU runtime (Step 1.3) | `edgetpu_2.0.0_all.deb` + `pip3 install pycoral` | **`libedgetpu1-std`** dari repo apt Google Coral, dan `pycoral==2.0.0` dengan `--extra-index-url https://google-coral.github.io/py-repo/` |
| `requirements_rpi.txt` | Dipasang apa adanya | **Keluarkan** `tensorflow`, `libcamera`, `picamera2` |
| Stream kamera | Channel `101` (main stream) | Channel **`102`** (sub stream) |
| Ukuran masukan | `img_size` 256 | **320** (`IMGSZ = 320`) |
| Konfigurasi | `config/camera_config.json`, `model_config.json` | Konstanta di awal **`main_app.py`** |
| Program inferensi | `deployment/tesrtspcoba.py` | **`main_app.py`** (dua proses, dashboard Flask, database) |
| Folder kerja | `/home/pi/conveyor-belt-damage-detection/` | `/home/pi/conveyor_project/` |
| Cek jaringan | `ping 8.8.8.8` | Sistem berjalan di **LAN tanpa internet**: ping ke IP kamera |
| Keluaran | Bounding box + FPS | Keputusan keparahan per sesi, dashboard, SQLite |

---

## 4. 🚀 Step 1: Setup Raspberry Pi

> **Ikuti `docs/setup_guide.md` wise78, Step 1.1 sampai 1.4** (Imager, pembaruan sistem, Edge TPU runtime, virtual environment), dengan penyesuaian di bagian 3 di atas.

### 1.1 Install Raspberry Pi OS

Flash **Raspberry Pi OS Bookworm 64-bit** dengan Raspberry Pi Imager; aktifkan SSH dan Wi-Fi/LAN.

### 1.2 Update sistem

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y build-essential git curl
```

Paket pengembangan lain (OpenCV, GStreamer, dll.) mengikuti daftar `setup_guide.md` wise78 bila dibutuhkan.

### 1.3 Python 3.9.18 lewat pyenv

```bash
# dependensi build pyenv
sudo apt install -y make libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
  libsqlite3-dev llvm libncursesw5-dev xz-utils tk-dev libxml2-dev \
  libxmlsec1-dev libffi-dev liblzma-dev

curl https://pyenv.run | bash        # lalu tambahkan inisialisasi pyenv ke ~/.bashrc
pyenv install 3.9.18
~/.pyenv/versions/3.9.18/bin/python -m venv ~/conveyor_env
source ~/conveyor_env/bin/activate
python --version                      # harus 3.9.18
```

### 1.4 Edge TPU runtime dan PyCoral

```bash
# repo apt Google Coral
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
  sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/coral.gpg
sudo apt update
sudo apt install -y libedgetpu1-std

# di dalam venv
pip install --upgrade pip
pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral==2.0.0
pip install tflite-runtime opencv-python numpy flask joblib scikit-learn scipy matplotlib
```

> Bila perintah penambahan repo di atas berbeda dengan petunjuk resmi Google Coral untuk Bookworm, ikuti petunjuk resmi Coral untuk bagian ini.

### 1.5 Verifikasi

```bash
lsusb | grep -i -E "1a6e|18d1"                 # Coral harus terdeteksi
python -c "from pycoral.utils import edgetpu; print(edgetpu.list_edge_tpus())"
```

> ⚠️ Jalankan skrip dari **folder induk**, jangan dari dalam folder hasil `git clone pycoral/` (lihat `TROUBLESHOOTING.md`).

---

## 5. 📷 Step 2: Setup IP Camera (RTSP)

> **Ikuti `docs/setup_guide.md` wise78, Step 2** (konfigurasi kamera dan uji koneksi RTSP).

### Ringkasan

1. Hubungkan kamera ke jaringan yang sama dengan Pi (disarankan kabel/PoE).
2. Cari IP kamera (SADP / HiTools Hikvision, atau halaman admin router).
3. Samakan subnet (contoh: `192.168.50.x`).
4. Atur nama pengguna/kata sandi dan aktifkan RTSP.
5. Uji aliran.

### Format URL RTSP Hikvision

```
rtsp://<USERNAME>:<PASSWORD>@<CAMERA_IP>:554/Streaming/Channels/<CHANNEL>
```

| Channel | Arti |
|---|---|
| `101` | Main stream (resolusi tinggi); dipakai contoh wise78 |
| `102` | Sub stream (resolusi lebih rendah); **dipakai proyek ini** |

### Uji RTSP

```bash
ping -c 3 192.168.50.2
ffmpeg -i "rtsp://admin:PASSWORD@192.168.50.2:554/Streaming/Channels/102" -t 10 test.mp4
```

```python
import cv2
cap = cv2.VideoCapture("rtsp://admin:PASSWORD@192.168.50.2:554/Streaming/Channels/102")
ok, frame = cap.read()
print("berhasil" if ok else "gagal", None if not ok else frame.shape)
```

### Menentukan ROI

ROI membatasi deteksi pada area belt dan menjadi dasar `R_max`.

```bash
python3 setup_roi.py     # butuh display (desktop Pi atau ssh -X)
```

Drag mouse mengelilingi area belt → **Enter/Space** → salin `[x1, y1, x2, y2]` ke `ROI_COORDS_AWAL` di `main_app.py`. ROI juga dapat diubah dari dashboard (`/set_roi`).

> ⚠️ **Keamanan:** jangan mengunggah `main_app.py` atau `setup_roi.py` yang berisi kata sandi kamera ke GitHub, laporan, atau grup.

---

## 6. 🧠 Step 3: Lapisan Deteksi YOLO

Seluruh tahap lapisan deteksi **bersumber dari repository wise78**. Ikuti repository tersebut:

| Tahap | Berkas di repository wise78 |
|---|---|
| Pengumpulan citra dari IP camera (ROI, filter ketajaman/kecerahan, anti-duplikat) | `data_collection/data_collector.py` |
| Konversi label polygon → bounding box | `data_collection/polygon_to_box.py` |
| Pelatihan YOLO11 di Google Colab | `training/train_yolo11_a100.ipynb` (`setup_guide.md` Step 4.1-4.2) |
| Konversi INT8 / Edge TPU | `training/convertedgetpu.py` (Step 4.3) |
| Transfer model ke Pi | `setup_guide.md` Step 5.1 |
| Inferensi real-time | `deployment/tesrtspcoba.py` |

### Spesifikasi yang dipakai proyek ini

| Item | Nilai |
|---|---|
| Model | YOLOv11n, INT8 penuh, kompilasi Edge TPU |
| Berkas | `best_full_integer_quant_edgetpu.tflite` |
| Lokasi di Pi | `/home/pi/conveyor_project/` |
| Ukuran masukan | 320 × 320 |
| Dataset | Roboflow `conveyor-belt-damage-s60ln` v5, 3.007 citra (1.233 asli), 4 kelas, CC BY 4.0 |
| Pembagian | Train 2.661 / Valid 231 / Test 115 |

> Dataset dan pelatihan YOLO diposisikan sebagai **adopsi** dari sumber publik, bukan kontribusi inti. Kontribusi inti ada di lapisan keputusan dan dashboard.

### Pelacakan objek (khas proyek ini)

Agar satu kerusakan yang tampak di banyak frame tidak dihitung berkali-kali, `main_app.py` memakai tracker berbasis IoU dan jarak. Hanya **objek baru** yang disimpan ke database. Tracker di-reset tiap sesi.

| Parameter | Nilai | Arti |
|---|---|---|
| `IOU_MIN_TRACKING` | 0,2 | IoU minimum agar dianggap objek yang sama |
| `TOLERANSI_JARAK_DASAR` | 20 piksel | Toleransi jarak dasar |
| `KECEPATAN_TRACKING_PIKSEL_PER_DETIK` | 150 | Tambahan toleransi sesuai gerak belt |
| `WAKTU_HILANG_MAX_DETIK` | 1,0 s | Objek dianggap hilang bila tak muncul selama ini |

---

## 7. ⚙️ Step 4: Konfigurasi Sistem

Berbeda dengan wise78 (JSON), konfigurasi proyek ini ada di awal `main_app.py`:

```python
# Model
MODEL_PATH     = "/home/pi/conveyor_project/best_full_integer_quant_edgetpu.tflite"
DEVICE         = "tpu:0"
IMGSZ          = 320
CONF_THRESHOLD = 0.25

# Kamera
CAMERA_IP = "192.168.50.2"
USERNAME  = "admin"
PASSWORD  = "ISI_PASSWORD_KAMERA"
CHANNEL   = "102"

# ROI
ENABLE_ROI      = True
ROI_COORDS_AWAL = [100, 100, 900, 600]

# Keputusan (isi dari hasil kalibrasi, Step 8)
W_R, W_N, W_C = 0.6, 0.1, 0.3
TAU   = 0.3221
N_MAX = 10

# Sesi, database, metode
DURASI_SESI_DETIK = 45
DB_PATH           = "conveyor_monitoring.db"
LOGREG_MODEL_PATH = "severity_logreg.pkl"
METODE_UTAMA      = "rule_based"      # atau "logreg" (hanya tampilan)
```

> Nilai `W_R`, `W_N`, `W_C`, `TAU`, `N_MAX`, dan `DURASI_SESI_DETIK` di atas hanya **contoh**. Gunakan nilai final dari kalibrasi dan pengukuran di lokasimu.

---

## 8. ⚖️ Step 5: Lapisan Keputusan (Tingkat Keparahan)

Dua metode dihitung dan disimpan **setiap sesi**. `METODE_UTAMA` hanya menentukan metode yang ditampilkan sebagai keputusan resmi di dashboard.

### Fitur sesi

```
R_max  = rasio luas bounding box terbesar terhadap luas ROI
N_norm = min(N / N_MAX, 1)
C_avg  = rata-rata confidence
```

### Metode utama: Rule-based SAW

```
S = W_R · R_max + W_N · N_norm + W_C · C_avg

N = 0       → Tidak Rusak
S > TAU     → Rusak Berat
selain itu  → Rusak Ringan
```

### Metode pembanding: Regresi Logistik

- Fitur `[R_max, N_norm, C_avg]`, dilatih oleh `train_logreg.py` → `severity_logreg.pkl`.
- Dimuat **sekali saat program dimulai**.
- Sesi `N = 0` tidak dijalankan lewat model.
- Hasil disimpan di `category_logreg` dan `prob_berat_logreg`.

> Parameter dan model **harus final sebelum data uji direkam**. Mengubahnya setelah melihat hasil uji membuat hasil menjadi optimistis.

---

## 9. 🗄️ Step 6: Database SQLite

Berkas `conveyor_monitoring.db` (modul `db_severity.py`, mode WAL).

### Tabel `sessions` (satu baris per sesi)

| Kolom | Isi |
|---|---|
| `session_id`, `timestamp`, `lokasi` | Identitas sesi |
| `n_detections` | Jumlah deteksi unik |
| `r_max`, `n_norm`, `c_avg`, `score_s` | Fitur dan skor |
| `category_system` | Kategori rule-based |
| `category_teknisi` | Label teknisi (acuan) |
| `category_logreg`, `prob_berat_logreg` | Hasil regresi logistik |
| `catatan` | Catatan |

### Tabel `detections` (satu baris per objek)

`session_id`, `class_name`, `bbox_area`, `roi_area`, `area_ratio`, `confidence`, `timestamp`.

### Memberi label teknisi

```python
from db_severity import update_label_teknisi
update_label_teknisi("S-0001", "Rusak Ringan", catatan="goresan permukaan")
```

> `update_label_teknisi` **menimpa** kolom `catatan`. Label diberikan **per sesi** oleh teknisi yang tidak melihat keputusan sistem.

---

## 10. ▶️ Step 7: Menjalankan Sistem dan Dashboard

```bash
cd /home/pi/conveyor_project
source ~/conveyor_env/bin/activate
python3 main_app.py
```

Buka dari perangkat lain di jaringan yang sama: `http://<IP-RASPBERRY-PI>:5000`. Letakkan `dashboard.html` di folder `templates/`.

### Endpoint

| Endpoint | Fungsi |
|---|---|
| `/` | Dashboard |
| `/video_feed` | Aliran video MJPEG |
| `/stream` | Data real-time (SSE) |
| `/start_session` | Memulai satu sesi |
| `/start_batch`, `/stop_batch` | Memulai / menghentikan Mode Custom Sesi |
| `/set_roi` | Mengubah ROI dari dashboard |

### Mode pada dashboard

| Mode | Kegunaan |
|---|---|
| Sesi tunggal | Satu sesi, satu keputusan |
| **Mode Custom Sesi** | Beberapa sesi berturut-turut (mis. 10 sesi × 30 detik untuk putaran 300 detik). Keputusan akhir = sesi dengan keparahan tertinggi (*worst-case*). **Mulai** memulai, **Stop** menghentikan |

### Aturan durasi sesi

- `DURASI_SESI_DETIK` harus **sama** untuk data latih, data uji, dan operasi.
- Satu sesi hanya mencakup sebagian belt, sehingga label teknisi juga **per sesi**.
- Untuk satu putaran penuh, jalankan sesi berturut-turut (Mode Custom Sesi).

Peringatan suhu muncul bila melewati `SUHU_PERINGATAN_C` (70 °C), sebelum *throttling* di sekitar 80-85 °C.

---

## 11. 🧪 Step 8: Pengumpulan Data, Kalibrasi, dan Pengujian

```
[1] Data latih → [2] Label teknisi → [3] Kalibrasi rule-based + latih logreg
        ↓
[4] Pasang parameter final & model di main_app.py
        ↓
[5] Rekam data uji → [6] Label teknisi (buta) → [7] Bandingkan metode
```

### Langkah 1-2: Data latih dan label

- Gunakan durasi sesi yang akan dipakai seterusnya.
- Minimal 30 sesi berlabel (ideal 60-80 atau lebih), Ringan dan Berat seimbang.
- Label per sesi. Pada fase ini `category_logreg` kosong dan keputusan memakai rule-based.

### Langkah 3: Kalibrasi dan pelatihan

```bash
python3 kalibrasi_rule_based.py     # mencari W_R, W_N, W_C, N_MAX, TAU
python3 train_logreg.py             # menghasilkan severity_logreg.pkl
```

Data dibagi 70:30 berstrata (`random_state=42`); parameter dicari pada 70%. Salin hasil ke `main_app.py`. **Latih ulang** regresi logistik bila `N_MAX` berubah.

### Langkah 4: Pemasangan

Letakkan `severity_logreg.pkl` di folder `main_app.py`, lalu jalankan ulang program. Mulai saat itu `category_logreg` dan `prob_berat_logreg` terisi otomatis, apa pun nilai `METODE_UTAMA`.

### Langkah 5-6: Data uji

Rekam sesi **setelah** model dimuat dengan konfigurasi yang sama. Teknisi melabeli per sesi tanpa melihat keputusan sistem. Catat batas waktu data latih dan data uji (database keduanya tampak sama).

### Langkah 7: Perbandingan

```bash
python3 bandingkan_metode.py --db conveyor_monitoring.db \
        --sejak "2026-10-20 00:00:00" --csv hasil.csv --grafik hasil.png
```

Keluaran: confusion matrix, precision, recall, F1, spesifisitas, accuracy, balanced accuracy, Cohen's Kappa (CI 95%), MCC, dan uji McNemar untuk skenario 3 kelas dan biner.

### Prinsip metodologi

- Jangan ubah parameter atau model setelah melihat hasil uji.
- Jangan membuang atau mengubah label setelah melihat hasil.
- Data uji terpisah dari data latih.
- Laporkan jumlah sesi dan interval kepercayaan, bukan satu angka Kappa saja.

---

## 12. 🖥️ Simulator Tanpa Raspberry Pi

`simulator_dashboard.py` menjalankan dashboard di komputer biasa dengan belt virtual (tanpa kamera, Coral, Pi).

```bash
pip install flask pillow
mkdir -p templates && cp dashboard.html templates/
python3 simulator_dashboard.py --kondisi campuran --durasi 30 --putaran 300 --kecepatan 10
```

| Opsi | Arti |
|---|---|
| `--kondisi` | `bersih`, `ringan`, `berat`, `campuran` |
| `--durasi` | Durasi sesi (detik) |
| `--putaran` | Waktu satu putaran belt (detik) |
| `--kecepatan` | Faktor percepatan simulasi |
| `--jeda`, `--seed`, `--port` | Jeda antar sesi, seed acak, port (bawaan 5000) |

`generate_synthetic_db.py` membuat database latih/uji sintetis untuk menguji program analisis:

```bash
python3 generate_synthetic_db.py --mode latih
python3 generate_synthetic_db.py --mode uji
python3 bandingkan_metode.py --db conveyor_monitoring_SYNTHETIC_UJI.db
```

> ⚠️ Data simulator dan data sintetis **bukan data penelitian** dan tidak boleh dilaporkan sebagai hasil.

---

## 13. ⚙️ Configuration Details

### Konfigurasi `main_app.py`

| Parameter | Arti |
|---|---|
| `MODEL_PATH`, `DEVICE`, `IMGSZ`, `CONF_THRESHOLD` | Model Edge TPU dan ambang confidence |
| `CAMERA_IP`, `USERNAME`, `PASSWORD`, `CHANNEL` | Akses kamera |
| `ENABLE_ROI`, `ROI_COORDS_AWAL` | ROI |
| `W_R, W_N, W_C`, `TAU`, `N_MAX` | Parameter rule-based (hasil kalibrasi) |
| `DURASI_SESI_DETIK` | Durasi satu sesi |
| `DB_PATH`, `LOGREG_MODEL_PATH` | Database dan model regresi logistik |
| `METODE_UTAMA` | `"rule_based"` atau `"logreg"` (tampilan saja) |
| `IOU_MIN_TRACKING`, `TOLERANSI_JARAK_DASAR`, `KECEPATAN_TRACKING_PIKSEL_PER_DETIK`, `WAKTU_HILANG_MAX_DETIK` | Tracker |
| `SUHU_PERINGATAN_C` | Ambang peringatan suhu |

### Parameter tahap pengumpulan data (milik repository wise78)

`config/camera_config.json` dipakai oleh `data_collector.py`: `rtsp_url`, `use_gstreamer`, `read_width`, `read_height`, `interval_sec`, `min_lap_var`, `min_bright`, `max_bright`. Penjelasan nilainya ada di `setup_guide.md` wise78.

---

## 14. 🔄 Maintenance

| Periode | Tugas |
|---|---|
| Harian (saat dipakai) | `vcgencmd measure_temp`, `df -h`, `free -h`; periksa dashboard dan video |
| Mingguan | `sudo apt update && sudo apt upgrade`; cadangkan database dan konfigurasi |
| Bulanan | Uji alur penuh; tinjau apakah parameter dan model masih sesuai kondisi belt |

Cadangkan juga `severity_logreg.pkl`, parameter hasil kalibrasi, dan berkas label teknisi. Jangan menyalin `.db` saat program berjalan (mode WAL punya berkas `-wal`/`-shm`).

---

## 15. 📁 Daftar Berkas

| Berkas | Fungsi |
|---|---|
| `main_app.py` | Program utama: kamera, deteksi, keputusan, dashboard |
| `templates/dashboard.html` | Tampilan dashboard |
| `db_severity.py` | Modul database SQLite |
| `setup_roi.py` | Menentukan ROI secara visual |
| `kalibrasi_rule_based.py` | Mencari parameter rule-based |
| `train_logreg.py` | Melatih regresi logistik |
| `bandingkan_metode.py` | Perbandingan lengkap kedua metode |
| `analisis_perbandingan.py` | Confusion matrix dan grafik untuk BAB 4.5 |
| `simulator_dashboard.py` | Simulator tanpa Raspberry Pi |
| `generate_synthetic_db.py` | Pembuat database sintetis |
| `severity_logreg.pkl` | Model regresi logistik |
| `best_full_integer_quant_edgetpu.tflite` | Model YOLOv11n Edge TPU |

---

## 📚 Sumber

- Repository rujukan: <https://github.com/wise78/Conveyor-Belt-Damage-Detection> (MIT License)
- Dataset: Roboflow, workspace `cataract-detection-n5ijc`, project `conveyor-belt-damage-s60ln`, versi 5 (CC BY 4.0)

**Catatan:** uji semua langkah di lingkungan aman sebelum dipakai di lapangan, dan cadangkan konfigurasi serta data penting sebelum mengubah apa pun.
