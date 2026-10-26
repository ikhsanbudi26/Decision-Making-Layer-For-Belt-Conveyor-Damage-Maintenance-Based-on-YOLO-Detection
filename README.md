# Decision Making Layer for Belt Conveyor Damage Maintenance Based on YOLO Detection

Sistem pendukung keputusan tingkat keparahan kerusakan *conveyor belt* berbasis **YOLOv11** dan **Edge AI**, dengan **dashboard pemantauan real-time**. Sistem berjalan di **Raspberry Pi 5** dengan **Google Coral Edge TPU**, memakai kamera IP **Hikvision (RTSP)**, dan mengubah hasil deteksi kerusakan menjadi satu keputusan yang mudah dipahami operator: **Tidak Rusak**, **Rusak Ringan**, atau **Rusak Berat**.

> Proyek ini dikembangkan sebagai bagian dari skripsi Teknik Elektro, berdasarkan kerja praktik di PT Indocement Tunggal Prakarsa Tbk (Site Cirebon). Lapisan deteksi (pengumpulan data, pelatihan YOLO, konversi Edge TPU) bersumber dari repository [wise78/Conveyor-Belt-Damage-Detection](https://github.com/wise78/Conveyor-Belt-Damage-Detection). **Kontribusi proyek ini ada pada lapisan keputusan, database, dan dashboard.**

---

## 📑 Daftar Isi

- [Deskripsi](#-deskripsi)
- [Fitur Utama](#-fitur-utama)
- [Arsitektur Sistem](#-arsitektur-sistem)
- [Lapisan Keputusan](#-lapisan-keputusan-kontribusi-utama)
- [Struktur Repository](#-struktur-repository)
- [Kebutuhan Sistem](#-kebutuhan-sistem)
- [Instalasi](#-instalasi)
- [Cara Menjalankan](#-cara-menjalankan)
- [Alur Pengumpulan Data, Kalibrasi, dan Pengujian](#-alur-pengumpulan-data-kalibrasi-dan-pengujian)
- [Konfigurasi](#-konfigurasi)
- [Database](#-database)
- [Dataset dan Model](#-dataset-dan-model)
- [Evaluasi dan Hasil](#-evaluasi-dan-hasil)
- [Keterbatasan](#-keterbatasan)
- [Troubleshooting](#-troubleshooting)
- [Dokumentasi Lain](#-dokumentasi-lain)
- [Keamanan](#-keamanan)
- [Kredit dan Lisensi](#-kredit-dan-lisensi)

---

## 🎯 Deskripsi

Model deteksi objek hanya menjawab "di mana kerusakan dan jenisnya apa". Keputusan perawatan membutuhkan jawaban yang lebih ringkas: **seberapa parah kondisi belt**. Repository ini menambahkan **lapisan pengambilan keputusan** di atas hasil deteksi YOLO:

1. Mendeteksi kerusakan secara real-time dari aliran kamera RTSP.
2. Melacak objek agar satu kerusakan tidak dihitung berulang.
3. Merangkum satu **sesi** pengamatan menjadi tiga fitur: jumlah kerusakan, luas kerusakan terbesar, dan rata-rata *confidence*.
4. Menghitung tingkat keparahan dengan dua metode: **rule-based (SAW berbobot)** sebagai metode utama dan **regresi logistik** sebagai pembanding.
5. Menampilkan hasil di dashboard web dan menyimpannya ke SQLite untuk analisis riwayat.
6. Membandingkan kedua metode terhadap **penilaian teknisi** memakai Cohen's Kappa, precision, recall, dan F1.

### Tiga kategori keputusan

| Kategori | Arti |
|---|---|
| ✅ **Tidak Rusak** | Tidak ada deteksi pada sesi tersebut |
| ⚠️ **Rusak Ringan** | Ada kerusakan, skor keparahan di bawah ambang |
| 🚨 **Rusak Berat** | Skor keparahan melewati ambang |

---

## ✨ Fitur Utama

- **Deteksi real-time** YOLOv11n INT8 pada Coral Edge TPU, masukan 320 × 320.
- **ROI (Region of Interest)** yang dapat diatur lewat alat bantu atau langsung dari dashboard.
- **Object tracking** ringan berbasis IoU dan jarak untuk mencegah duplikasi deteksi.
- **Sesi pengamatan** berdurasi tetap, dengan **Mode Custom Sesi** untuk menilai satu putaran belt penuh (keputusan akhir memakai prinsip *worst-case*).
- **Dua metode keputusan** yang dihitung bersamaan: rule-based SAW dan regresi logistik.
- **Dashboard real-time** (Flask, SSE, MJPEG) di jaringan lokal tanpa internet.
- **Database SQLite** (mode WAL) berisi ringkasan sesi, detail deteksi, dan label teknisi.
- **Program kalibrasi dan evaluasi**: pencarian parameter, pelatihan regresi logistik, dan perbandingan metode lengkap.
- **Simulator dashboard** tanpa Raspberry Pi, dan **generator data sintetis** untuk menguji program analisis.
- **Peringatan suhu** Raspberry Pi sebelum *thermal throttling*.

---

## 🏗️ Arsitektur Sistem

### Empat lapisan

```
┌──────────────┐   ┌────────────────┐   ┌─────────────────────┐   ┌────────────────┐
│ 1. KAMERA    │ → │ 2. DETEKSI     │ → │ 3. KEPUTUSAN        │ → │ 4. DASHBOARD   │
│ IP Camera    │   │ YOLOv11n INT8  │   │ Fitur sesi:         │   │ Flask + SSE    │
│ Hikvision    │   │ Coral Edge TPU │   │  N, R_max, C_avg    │   │ MJPEG video    │
│ RTSP (TCP)   │   │ ROI + Tracker  │   │ Rule-based SAW      │   │ SQLite         │
│              │   │                │   │ Regresi Logistik    │   │ (riwayat)      │
└──────────────┘   └────────────────┘   └─────────────────────┘   └────────────────┘
```

### Arsitektur proses

`main_app.py` berjalan sebagai **dua proses terpisah** (spawn) agar pembacaan kamera dan inferensi tidak terblokir oleh server web:

```
┌─────────────────────────────┐        ┌─────────────────────────────────┐
│ Proses 1: Deteksi           │        │ Proses 2: Server Web (Flask)    │
│ - baca RTSP                 │ frame  │ - /            dashboard        │
│ - ROI + inferensi Edge TPU  │ ─────► │ - /video_feed  MJPEG            │
│ - tracking                  │ (RAM,  │ - /stream      SSE (data live)  │
│ - hitung fitur sesi         │/dev/shm)│ - /start_session, /start_batch │
│ - keputusan + simpan SQLite │ ◄───── │ - /stop_batch, /set_roi         │
└─────────────────────────────┘ perintah└─────────────────────────────────┘
```

Frame sementara ditulis ke `/dev/shm` (RAM), bukan kartu SD, sehingga aman ditulis berkali-kali per detik.

### Alur satu sesi

```
Mulai sesi → [loop frame] RTSP → ROI → YOLO (Edge TPU) → tracking → objek baru disimpan
          → durasi sesi habis → hitung N, R_max, N_norm, C_avg
          → S = W_R·R_max + W_N·N_norm + W_C·C_avg
          → kategori rule-based  (+ kategori & peluang regresi logistik bila model dimuat)
          → simpan ke SQLite → kirim ke dashboard (SSE)
```

### Mode Custom Sesi (satu putaran belt)

Satu sesi hanya mencakup segmen belt yang terlihat selama durasi sesi. Untuk menilai seluruh belt, jalankan beberapa sesi berturut-turut. **Keputusan akhir mengikuti sesi dengan keparahan tertinggi** (*worst-case*):

```
Tidak Rusak  <  Rusak Ringan  <  Rusak Berat
```

Contoh: putaran belt 300 detik dan durasi sesi 30 detik → 10 sesi; bila satu sesi berkategori Rusak Berat, keputusan akhir adalah Rusak Berat.

---

## ⚖️ Lapisan Keputusan (Kontribusi Utama)

### Fitur per sesi

| Simbol | Rumus / arti |
|---|---|
| `N` | Jumlah kerusakan unik setelah tracking |
| `R_max` | Rasio luas bounding box terbesar terhadap luas ROI |
| `N_norm` | `min(N / N_MAX, 1)` |
| `C_avg` | Rata-rata *confidence* deteksi |

### Metode utama: Rule-based SAW (*Simple Additive Weighting*)

```
S = W_R · R_max + W_N · N_norm + W_C · C_avg

N = 0       →  Tidak Rusak
S > TAU     →  Rusak Berat
selain itu  →  Rusak Ringan
```

Bobot `W_R, W_N, W_C`, ambang `TAU`, dan `N_MAX` dicari lewat **kalibrasi** terhadap penilaian teknisi (grid search bobot dan `N_MAX`, ambang `TAU` dicari eksak, tujuan memaksimalkan Cohen's Kappa pada data latih).

### Metode pembanding: Regresi Logistik

- Fitur: `[R_max, N_norm, C_avg]`.
- Dilatih oleh `train_logreg.py` → `severity_logreg.pkl` (ambang probabilitas 0,5).
- Dimuat sekali saat program dimulai; hasilnya (`category_logreg`, `prob_berat_logreg`) **selalu disimpan** ke database.
- `METODE_UTAMA` hanya menentukan metode yang **ditampilkan** sebagai keputusan resmi di dashboard. Kedua metode tetap dihitung.

### Mengapa dua metode?

Rule-based mudah dijelaskan dan diaudit (setiap bobot punya makna). Regresi logistik belajar dari data dan menjadi pembanding objektif. Keduanya dievaluasi pada **data uji yang sama** terhadap label teknisi.

---

## 📁 Struktur Repository

> Sesuaikan dengan susunan folder sebenarnya di repository. Struktur berikut adalah susunan yang disarankan.

```
Decision-Making-Layer-For-Belt-Conveyor-Damage-Maintenance-Based-on-YOLO-Detection/
├── README.md                          # Dokumentasi utama (berkas ini)
├── requirements.txt                   # Dependensi Python (Raspberry Pi)
├── .gitignore                         # Mengecualikan database, kata sandi, model besar
│
├── main_app.py                        # Program utama: kamera → deteksi → keputusan → dashboard
├── db_severity.py                     # Modul database SQLite (sessions, detections, label)
├── setup_roi.py                       # Alat bantu menentukan ROI secara visual
│
├── templates/
│   └── dashboard.html                 # Tampilan dashboard (dilayani Flask)
│
├── analysis/                          # Kalibrasi dan evaluasi
│   ├── kalibrasi_rule_based.py        # Mencari W_R, W_N, W_C, N_MAX, TAU
│   ├── train_logreg.py                # Melatih regresi logistik → severity_logreg.pkl
│   ├── bandingkan_metode.py           # Perbandingan lengkap (Kappa, P/R/F1, MCC, McNemar)
│   └── analisis_perbandingan.py       # Confusion matrix dan grafik untuk laporan
│
├── tools/                             # Alat bantu pengujian (bukan data penelitian)
│   ├── simulator_dashboard.py         # Simulator dashboard tanpa Raspberry Pi
│   └── generate_synthetic_db.py       # Pembuat database sintetis (latih/uji)
│
├── models/                            # Berkas model (lihat catatan ukuran)
│   ├── best_full_integer_quant_edgetpu.tflite   # YOLOv11n INT8 Edge TPU
│   └── severity_logreg.pkl                       # Hasil train_logreg.py
│
├── docs/
│   ├── GUIDEBOOK.md                   # Panduan instalasi dan penggunaan lengkap
│   └── TROUBLESHOOTING.md             # Panduan penyelesaian masalah
│
└── LICENSE
```

**Rujukan lapisan deteksi** (tidak disalin ke repository ini): pengumpulan data (`data_collector.py`), konversi label (`polygon_to_box.py`), pelatihan (`train_yolo11_a100.ipynb`), dan konversi Edge TPU (`convertedgetpu.py`) mengikuti [repository wise78](https://github.com/wise78/Conveyor-Belt-Damage-Detection).

### Deskripsi berkas

| Berkas | Fungsi |
|---|---|
| `main_app.py` | Dua proses (deteksi dan server web), tracker, sesi, mode batch, keputusan, penyimpanan database |
| `db_severity.py` | Membuat tabel, menyimpan sesi dan deteksi, mengambil data kalibrasi, memberi label teknisi |
| `setup_roi.py` | Menggambar ROI dengan mouse dari satu frame RTSP |
| `templates/dashboard.html` | UI: video langsung, kategori keputusan, metrik sesi, Mode Custom Sesi, riwayat |
| `kalibrasi_rule_based.py` | Grid search parameter rule-based dengan pemisahan 70:30 |
| `train_logreg.py` | Melatih dan menyimpan regresi logistik |
| `bandingkan_metode.py` | Evaluasi dua skenario (3 kelas dan biner) dengan interval kepercayaan Kappa |
| `analisis_perbandingan.py` | Confusion matrix dan grafik perbandingan |
| `simulator_dashboard.py` | Belt virtual untuk mencoba dashboard tanpa perangkat keras |
| `generate_synthetic_db.py` | Data sintetis untuk menguji program analisis |

---

## 🧰 Kebutuhan Sistem

### Hardware

| Komponen | Keterangan |
|---|---|
| Raspberry Pi 5 | 4 GB minimum, 8 GB disarankan; catu daya 5 V / 5 A |
| Google Coral USB Accelerator | Edge TPU |
| Kamera IP Hikvision | RTSP, disarankan PoE |
| MicroSD | ≥ 32 GB, Class 10 / A1 |
| Pendingin | Heatsink dan kipas |
| Jaringan | Ethernet/switch; Pi dan kamera satu subnet |

### Software

| Komponen | Versi |
|---|---|
| Raspberry Pi OS | Bookworm 64-bit |
| Python | **3.9.18** (lewat `pyenv`, venv `conveyor_env`) |
| Edge TPU runtime | `libedgetpu1-std` |
| PyCoral | `pycoral 2.0.0` |
| Pustaka | `tflite-runtime`, `opencv-python`, `numpy`, `flask`, `joblib`, `scikit-learn`, `scipy`, `matplotlib` |

---

## 🚀 Instalasi

Panduan lengkap: [`docs/GUIDEBOOK.md`](docs/GUIDEBOOK.md). Ringkasan:

```bash
# 1. Python 3.9.18 lewat pyenv dan virtual environment
pyenv install 3.9.18
~/.pyenv/versions/3.9.18/bin/python -m venv ~/conveyor_env
source ~/conveyor_env/bin/activate

# 2. Edge TPU runtime (apt, repo Google Coral) dan PyCoral
sudo apt install -y libedgetpu1-std
pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral==2.0.0

# 3. Dependensi lain
pip install tflite-runtime opencv-python numpy flask joblib scikit-learn scipy matplotlib

# 4. Ambil repository
git clone https://github.com/ikhsanbudi26/Decision-Making-Layer-For-Belt-Conveyor-Damage-Maintenance-Based-on-YOLO-Detection.git
cd Decision-Making-Layer-For-Belt-Conveyor-Damage-Maintenance-Based-on-YOLO-Detection
```

Instalasi Raspberry Pi, setup kamera RTSP, dan seluruh tahap lapisan deteksi YOLO mengikuti [repository wise78](https://github.com/wise78/Conveyor-Belt-Damage-Detection) (`docs/setup_guide.md`), dengan penyesuaian untuk Pi 5 yang dijelaskan di `GUIDEBOOK.md`.

> ⚠️ Jalankan skrip dari folder proyek, **bukan** dari dalam folder hasil `git clone pycoral/`.

---

## ▶️ Cara Menjalankan

### 1. Atur konfigurasi dan ROI

Edit konstanta di awal `main_app.py` (lihat [Konfigurasi](#-konfigurasi)), lalu tentukan ROI:

```bash
python3 setup_roi.py        # butuh display (desktop Pi atau ssh -X)
```

### 2. Jalankan sistem

```bash
python3 main_app.py
```

### 3. Buka dashboard

Dari perangkat lain di jaringan yang sama:

```
http://<IP-RASPBERRY-PI>:5000
```

Letakkan `dashboard.html` di folder `templates/`.

### 4. Coba tanpa Raspberry Pi (simulator)

```bash
pip install flask pillow
python3 tools/simulator_dashboard.py --kondisi campuran --durasi 30 --putaran 300 --kecepatan 10
```

| Opsi | Arti |
|---|---|
| `--kondisi` | `bersih`, `ringan`, `berat`, `campuran` |
| `--durasi` | Durasi sesi (detik) |
| `--putaran` | Waktu satu putaran belt (detik) |
| `--kecepatan` | Faktor percepatan simulasi |
| `--port` | Port dashboard (bawaan 5000) |

---

## 🧪 Alur Pengumpulan Data, Kalibrasi, dan Pengujian

```
[1] Data latih → [2] Label teknisi → [3] Kalibrasi rule-based + latih logreg
        ↓
[4] Pasang parameter final & model di main_app.py
        ↓
[5] Rekam data uji → [6] Label teknisi (buta) → [7] Bandingkan metode
```

```bash
# Langkah 3
python3 analysis/kalibrasi_rule_based.py     # parameter rule-based
python3 analysis/train_logreg.py             # severity_logreg.pkl

# Langkah 7
python3 analysis/bandingkan_metode.py --db conveyor_monitoring.db \
        --sejak "2026-10-20 00:00:00" --csv hasil.csv --grafik hasil.png
```

### Prinsip metodologi

- **Durasi sesi sama** untuk data latih, data uji, dan operasi.
- **Label teknisi per sesi** (segmen belt yang terlihat), diberikan tanpa melihat keputusan sistem.
- **Parameter dan model final sebelum data uji direkam**; jangan diubah setelah melihat hasil uji.
- **Data latih dan data uji terpisah** (batas timestamp atau database terpisah).
- Laporkan **jumlah sesi dan interval kepercayaan** Cohen's Kappa, bukan satu angka saja.

### Metrik yang dihitung `bandingkan_metode.py`

Confusion matrix, precision, recall, F1 (per kelas, macro, weighted), spesifisitas, accuracy, balanced accuracy, Cohen's Kappa (biasa dan *quadratic*, dengan CI 95% bootstrap), MCC, dan uji McNemar. Evaluasi dilakukan pada dua skenario: **3 kelas** (semua sesi berlabel) dan **biner** Ringan vs Berat (sesi dengan deteksi).

---

## ⚙️ Konfigurasi

Seluruh konfigurasi ada di awal `main_app.py`:

```python
# Model
MODEL_PATH     = "/home/pi/conveyor_project/best_full_integer_quant_edgetpu.tflite"
DEVICE         = "tpu:0"
IMGSZ          = 320
CONF_THRESHOLD = 0.25

# Kamera (jangan unggah kata sandi asli)
CAMERA_IP = "192.168.50.2"
USERNAME  = "admin"
PASSWORD  = "ISI_PASSWORD_KAMERA"
CHANNEL   = "102"            # 101 = main stream, 102 = sub stream

# ROI
ENABLE_ROI      = True
ROI_COORDS_AWAL = [100, 100, 900, 600]

# Keputusan (isi dari hasil kalibrasi)
W_R, W_N, W_C = 0.6, 0.1, 0.3
TAU   = 0.3221
N_MAX = 10

# Sesi, database, metode
DURASI_SESI_DETIK = 45
DB_PATH           = "conveyor_monitoring.db"
LOGREG_MODEL_PATH = "severity_logreg.pkl"
METODE_UTAMA      = "rule_based"      # atau "logreg" (hanya tampilan)
```

> Nilai `W_R`, `W_N`, `W_C`, `TAU`, `N_MAX`, dan `DURASI_SESI_DETIK` di atas adalah **contoh**; gunakan hasil kalibrasi dan pengukuran di lokasimu.

### Parameter tracker

| Parameter | Nilai | Arti |
|---|---|---|
| `IOU_MIN_TRACKING` | 0,2 | IoU minimum agar dianggap objek yang sama |
| `TOLERANSI_JARAK_DASAR` | 20 px | Toleransi jarak dasar |
| `KECEPATAN_TRACKING_PIKSEL_PER_DETIK` | 150 | Tambahan toleransi sesuai gerak belt |
| `WAKTU_HILANG_MAX_DETIK` | 1,0 s | Objek dianggap hilang bila tak muncul selama ini |

Tracker di-reset setiap sesi; hanya objek baru yang disimpan ke database.

---

## 🗄️ Database

`conveyor_monitoring.db` (SQLite, mode WAL).

### Tabel `sessions` (satu baris per sesi)

| Kolom | Isi |
|---|---|
| `session_id`, `timestamp`, `lokasi` | Identitas sesi |
| `n_detections` | Jumlah deteksi unik |
| `r_max`, `n_norm`, `c_avg`, `score_s` | Fitur dan skor |
| `category_system` | Kategori rule-based |
| `category_teknisi` | Label teknisi (acuan evaluasi) |
| `category_logreg`, `prob_berat_logreg` | Hasil regresi logistik |
| `catatan` | Catatan |

### Tabel `detections` (satu baris per objek)

`session_id`, `class_name`, `bbox_area`, `roi_area`, `area_ratio`, `confidence`, `timestamp`.

### Memberi label teknisi

```python
from db_severity import update_label_teknisi
update_label_teknisi("S-0001", "Rusak Ringan", catatan="goresan permukaan")
```

> `update_label_teknisi` menimpa kolom `catatan`. Salin database hanya saat program berhenti (mode WAL membuat berkas `-wal` dan `-shm`).

---

## 📦 Dataset dan Model

| Item | Keterangan |
|---|---|
| Model | YOLOv11n, kuantisasi INT8 penuh, kompilasi Edge TPU |
| Kelas | `Hole`, `Puncture`, `Tear`, `Patch Work` |
| Dataset | Roboflow `conveyor-belt-damage-s60ln` versi 5 (workspace `cataract-detection-n5ijc`) |
| Jumlah citra | 3.007 (1.233 asli, sisanya hasil augmentasi) |
| Pembagian | Train 2.661 / Valid 231 / Test 115 |
| Ukuran masukan | 320 × 320 |
| Lisensi dataset | CC BY 4.0 |

Dataset dan pelatihan YOLO diposisikan sebagai **adopsi** dari sumber publik, bukan kontribusi inti proyek ini. Prosedur pelatihan dan konversi mengikuti repository wise78.

---

## 📊 Evaluasi dan Hasil

> Isi bagian ini dengan hasil **data lapangan nyata** setelah pengujian selesai. Jangan memakai data sintetis atau simulator sebagai hasil penelitian.

| Metrik (data uji lapangan) | Rule-based | Regresi Logistik |
|---|---|---|
| Jumlah sesi uji | _isi_ | _isi_ |
| Accuracy | _isi_ | _isi_ |
| Precision (macro) | _isi_ | _isi_ |
| Recall (macro) | _isi_ | _isi_ |
| F1 (macro) | _isi_ | _isi_ |
| Cohen's Kappa (CI 95%) | _isi_ | _isi_ |
| MCC | _isi_ | _isi_ |

Sertakan juga confusion matrix, tangkapan layar dashboard, dan hasil `bandingkan_metode.py`.

---

## ⚠️ Keterbatasan

- Satu sesi hanya menilai **segmen belt yang terlihat**; keputusan seluruh belt bergantung pada Mode Custom Sesi dan durasi sesi yang dipilih.
- Fitur (`N`, `R_max`) bergantung pada durasi sesi, sehingga parameter hasil kalibrasi hanya berlaku untuk durasi yang sama.
- Jumlah data uji yang kecil membuat Kappa memiliki rentang ketidakpastian lebar.
- Kerusakan yang tidak terdeteksi YOLO (sesi `N = 0`) langsung dinilai "Tidak Rusak".
- Pelabelan teknisi bersifat subjektif; kualitas label membatasi kualitas evaluasi.
- Dataset YOLO berasal dari sumber publik, bukan dataset lapangan sendiri.
- Dashboard hanya untuk jaringan lokal (LAN); belum ada autentikasi pengguna.

---

## 🔧 Troubleshooting

Masalah umum dan solusinya ada di [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md), antara lain:

| Masalah | Lihat bagian |
|---|---|
| `No module named 'pycoral.pybind'` | Python dan Dependensi |
| Coral tidak terdeteksi | Edge TPU (Coral) |
| RTSP gagal / gambar berantakan | Kamera dan RTSP |
| Dashboard tidak terbuka | Jaringan dan Akses Dashboard |
| N membengkak / deteksi ganda | Deteksi dan Tracking |
| Kolom logreg bernilai NULL | Database |
| Kappa rendah / hasil sempurna | Kalibrasi dan Perbandingan Metode |

---

## 📚 Dokumentasi Lain

- [`docs/GUIDEBOOK.md`](docs/GUIDEBOOK.md): instalasi, konfigurasi, penggunaan, dan alur kalibrasi lengkap
- [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md): penyelesaian masalah
- [wise78/Conveyor-Belt-Damage-Detection](https://github.com/wise78/Conveyor-Belt-Damage-Detection): sumber lapisan deteksi (`docs/setup_guide.md`, `docs/troubleshooting.md`)

---

## 🔒 Keamanan

- **Jangan mengunggah kata sandi kamera** ke GitHub. Gunakan variabel lingkungan atau berkas konfigurasi yang dikecualikan lewat `.gitignore`.
- Jangan mengunggah database berisi data operasional yang sensitif.
- Dashboard tidak memiliki autentikasi; jalankan hanya di jaringan lokal tepercaya.

Contoh `.gitignore`:

```
*.db
*.db-wal
*.db-shm
__pycache__/
.env
config_rahasia.py
run.log
```

---

## 🙏 Kredit dan Lisensi

- **Lapisan deteksi dan prosedur dasar:** [wise78/Conveyor-Belt-Damage-Detection](https://github.com/wise78/Conveyor-Belt-Damage-Detection) (MIT License).
- **Dataset:** Roboflow, workspace `cataract-detection-n5ijc`, project `conveyor-belt-damage-s60ln`, versi 5 (CC BY 4.0).
- **YOLO:** [Ultralytics](https://github.com/ultralytics/ultralytics); periksa ketentuan lisensinya sebelum penggunaan komersial.
- **Edge TPU:** Google Coral.
- **Lokasi kerja praktik:** PT Indocement Tunggal Prakarsa Tbk, Site Cirebon.

**Lisensi:** tentukan lisensi repository ini (mis. MIT) dan tambahkan berkas `LICENSE`.

---

## 📞 Kontak

Pemilik repository: [@ikhsanbudi26](https://github.com/ikhsanbudi26). Untuk pertanyaan dan masukan, buat *issue* di GitHub.

---

**Catatan:** uji sistem di lingkungan aman sebelum dipakai di lapangan, dan cadangkan konfigurasi serta data penting sebelum mengubah apa pun.
