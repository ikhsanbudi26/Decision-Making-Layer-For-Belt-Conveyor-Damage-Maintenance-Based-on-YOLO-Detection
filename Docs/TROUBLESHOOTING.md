# 🔧 Troubleshooting Guide - Sistem Keputusan Keparahan Kerusakan Conveyor Belt

Panduan mengatasi masalah umum pada proyek ini (Raspberry Pi 5 + Coral Edge TPU + kamera Hikvision RTSP + dashboard Flask). Panduan instalasi dan penggunaan: [`GUIDEBOOK.md`](GUIDEBOOK.md).

> **Masalah yang prosedurnya sama dengan proyek wise78** (RTSP, Edge TPU, performa, pengumpulan data, konversi model, pelatihan YOLO) dibahas lengkap di `docs/troubleshooting.md` pada <https://github.com/wise78/Conveyor-Belt-Damage-Detection>. Di sini hanya dicantumkan ringkasan, **penyesuaian untuk Pi 5 / Bookworm**, dan masalah yang **khas proyek ini**: tracker, sesi, database, kalibrasi, dan perbandingan metode.

## 📑 Daftar Isi

1. [Quick Diagnosis](#1--quick-diagnosis)
2. [Python dan Dependensi](#2--python-dan-dependensi)
3. [Edge TPU (Coral)](#3--edge-tpu-coral)
4. [Model YOLO](#4--model-yolo)
5. [Kamera dan RTSP](#5--kamera-dan-rtsp)
6. [Jaringan dan Akses Dashboard](#6--jaringan-dan-akses-dashboard)
7. [Deteksi dan Tracking](#7--deteksi-dan-tracking)
8. [Sesi, Mode Custom Sesi, dan Dashboard](#8--sesi-mode-custom-sesi-dan-dashboard)
9. [Database](#9--database)
10. [Kalibrasi, Regresi Logistik, dan Perbandingan Metode](#10--kalibrasi-regresi-logistik-dan-perbandingan-metode)
11. [Performa dan Suhu](#11--performa-dan-suhu)
12. [Simulator dan Data Sintetis](#12--simulator-dan-data-sintetis)
13. [Advanced Troubleshooting](#13--advanced-troubleshooting)
14. [Monitoring dan Pemulihan](#14--monitoring-dan-pemulihan)
15. [Getting Help](#15--getting-help)
16. [Checklist Sebelum Pengambilan Data](#16--checklist-sebelum-pengambilan-data)

---

## 1. 🚨 Quick Diagnosis

Periksa dari lapisan bawah ke atas dan berhenti di lapisan pertama yang gagal.

### System Health Check

```bash
#!/bin/bash
# health_check.sh - pemeriksaan cepat (tanpa internet)

CAMERA_IP="192.168.50.2"

echo "=== System Health Check ==="
echo "1. CPU Temperature : $(vcgencmd measure_temp)"
echo "2. Throttling      : $(vcgencmd get_throttled)   (0x0 = normal)"
echo "3. Memory Usage    : $(free -h | grep Mem | awk '{print $3"/"$2}')"
echo "4. Disk Usage      : $(df -h / | tail -1 | awk '{print $5}')"
echo "5. Camera Ping     : $(ping -c 1 -W 2 $CAMERA_IP >/dev/null && echo 'OK' || echo 'FAIL')"
echo "6. Edge TPU (USB)  : $(lsusb | grep -i -E '1a6e|18d1' >/dev/null && echo 'OK' || echo 'NOT FOUND')"
echo "7. Python          : $(python3 --version 2>/dev/null || echo 'NOT FOUND')   (harus 3.9.x di venv)"
echo "8. OpenCV          : $(python3 -c 'import cv2; print(cv2.__version__)' 2>/dev/null || echo 'NOT FOUND')"
echo "9. PyCoral         : $(python3 -c 'from pycoral.utils import edgetpu; print(len(edgetpu.list_edge_tpus()), "TPU")' 2>/dev/null || echo 'NOT FOUND')"
echo "10. Dashboard      : $(curl -s -o /dev/null -w '%{http_code}' http://localhost:5000 2>/dev/null)   (200 = hidup)"
```

> Berbeda dengan skrip wise78 yang melakukan `ping 8.8.8.8`, sistem ini berjalan di **LAN tanpa internet**, jadi yang diuji adalah ping ke kamera. Jalankan skrip di dalam venv `conveyor_env`.

| Urutan | Pemeriksaan | Bila gagal, lihat |
|---|---|---|
| 1 | Suhu / throttling | [11](#11--performa-dan-suhu) |
| 2 | Coral terdeteksi | [3](#3--edge-tpu-coral) |
| 3 | Python 3.9 di venv, PyCoral | [2](#2--python-dan-dependensi) |
| 4 | Kamera dan RTSP | [5](#5--kamera-dan-rtsp) |
| 5 | Dashboard | [6](#6--jaringan-dan-akses-dashboard), [8](#8--sesi-mode-custom-sesi-dan-dashboard) |

---

## 2. 🐍 Python dan Dependensi

### Problem: `ModuleNotFoundError: No module named 'pycoral.pybind'`

**Penyebab:** skrip dijalankan dari dalam folder hasil `git clone pycoral/`, sehingga folder itu menutupi paket `pycoral` yang terpasang.

**Solusi:** pindah ke folder induk.

```bash
cd /home/pi/conveyor_project
python3 main_app.py
```

### Problem: `pip install pycoral` gagal / versi tidak ditemukan

**Penyebab:** `pycoral` tidak ada di PyPI standar dan tidak tersedia untuk Python 3.11 (bawaan Bookworm). Perintah `pip3 install pycoral` pada `setup_guide.md` wise78 tidak cukup di lingkungan ini.

```bash
source ~/conveyor_env/bin/activate
python --version        # harus 3.9.18
pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral==2.0.0
```

### Problem: `sudo apt install python3.9` tidak ditemukan

Di Bookworm, Python 3.9 tidak tersedia lewat apt. Pasang dengan `pyenv`:

```bash
pyenv install 3.9.18
~/.pyenv/versions/3.9.18/bin/python -m venv ~/conveyor_env
source ~/conveyor_env/bin/activate
```

### Problem: Python di venv bukan 3.9

Venv dibuat dengan Python sistem atau belum diaktifkan. Buat ulang dengan perintah di atas dan `source ~/conveyor_env/bin/activate`.

### Problem: `tensorflow`, `libcamera`, `picamera2` gagal dari `requirements_rpi.txt`

Keluarkan ketiganya dari berkas. Proyek ini memakai `tflite-runtime` dan kamera IP, bukan kamera CSI.

### Problem: `ImportError: libGL.so.1` / masalah OpenCV

```bash
sudo apt install -y libgl1 libglib2.0-0
# atau tanpa GUI
pip uninstall -y opencv-python && pip install opencv-python-headless
```

`setup_roi.py` butuh GUI, jadi memakai `opencv-python` biasa.

### Problem: `ModuleNotFoundError: sklearn / scipy / joblib / matplotlib`

```bash
pip install scikit-learn scipy joblib matplotlib
```

### Problem: `ModuleNotFoundError: db_severity`

Skrip analisis dijalankan dari folder yang tidak berisi `db_severity.py`. Jalankan dari folder yang sama.

---

## 3. 🔌 Edge TPU (Coral)

### Problem: Edge TPU tidak terdeteksi

```
Error: No Edge TPU device found
```

**A. Periksa USB**

```bash
lsusb | grep -i -E "1a6e|18d1"
```

Coral baru yang belum diinisialisasi muncul sebagai `1a6e:089a` (Global Unichip), lalu menjadi `18d1:9302` (Google) setelah dipakai pertama kali. Jadi `lsusb | grep Google` seperti di wise78 bisa gagal padahal Coral ada; pakai kedua ID di atas.

**B. Periksa runtime**

```bash
dpkg -l | grep edgetpu
sudo apt install libedgetpu1-std
```

Proyek ini memakai `libedgetpu1-std`. Perintah `apt remove/install edgetpu` pada panduan wise78 tidak berlaku. `libedgetpu1-max` (lebih cepat tetapi lebih panas) tidak dipasang bersamaan dengan `std`.

**C. Daya dan port**

- Pakai port USB 3.0 (biru) dan kabel bawaan.
- Pi 5 butuh catu daya 5 V / 5 A; daya kurang sering membuat Coral terputus.
- Hindari hub USB tanpa daya. Cabut dan colok ulang setelah memasang runtime.

**D. Izin pengguna**

```bash
sudo usermod -aG plugdev $USER      # lalu logout / login ulang
```

### Problem: `ValueError: Failed to load delegate from libedgetpu.so.1`

Runtime belum terpasang, Coral tidak terdeteksi, atau izin USB kurang. Ikuti A-D di atas.

### Problem: Model gagal dimuat

```python
import tflite_runtime.interpreter as tflite
interpreter = tflite.Interpreter(
    model_path="/home/pi/conveyor_project/best_full_integer_quant_edgetpu.tflite",
    experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
```

Bila gagal: periksa jalur model, pastikan berkas berakhiran `_edgetpu.tflite`, dan lakukan konversi ulang (lihat bagian 4).

---

## 4. 🧠 Model YOLO

### Problem: `FileNotFoundError` untuk model `.tflite`

```python
MODEL_PATH = "/home/pi/conveyor_project/best_full_integer_quant_edgetpu.tflite"
```

```bash
ls -l /home/pi/conveyor_project/*.tflite
```

### Problem: model berjalan lambat (tidak memakai Edge TPU)

Model yang dipakai bukan versi `*_edgetpu.tflite` hasil kompilasi Edge TPU, sehingga berjalan di CPU. Pakai berkas hasil konversi INT8 penuh.

### Problem: deteksi kosong atau sembarang setelah konversi

| Kemungkinan | Solusi |
|---|---|
| `IMGSZ` tidak sama dengan masukan model | Samakan (proyek ini 320; wise78 memakai 256) |
| Data kalibrasi kuantisasi tidak mewakili | Ulangi konversi dengan citra representatif |
| `CONF_THRESHOLD` terlalu tinggi | Turunkan bertahap (mis. 0,25) |
| Urutan kelas berbeda | Samakan dengan label saat pelatihan: Hole 0, Puncture 1, Tear 2, Patch Work 3 |

### Problem: konversi model / pelatihan gagal atau akurasi buruk

Lihat bagian *Training Issues* pada `docs/troubleshooting.md` wise78 (versi `ultralytics==8.3.185`, bentuk tensor, kualitas dataset, hiperparameter, *transfer learning*).

### Problem: banyak deteksi palsu

- Naikkan `CONF_THRESHOLD`.
- Persempit ROI agar tepi dan latar tidak ikut.
- Tambahkan contoh negatif (latar belt tanpa kerusakan) saat pelatihan ulang.
- `Patch Work` (tambalan lama) dapat terhitung sebagai kerusakan; perhatikan saat melabeli.

---

## 5. 📷 Kamera dan RTSP

### Problem: tidak bisa terhubung ke IP camera

```
Error: Failed to connect to RTSP stream
```

Periksa berurutan:

```bash
ping 192.168.50.2
nc -zv 192.168.50.2 554                       # port RTSP terbuka?
ffprobe "rtsp://admin:PASSWORD@192.168.50.2:554/Streaming/Channels/102"
ffmpeg -i "rtsp://admin:PASSWORD@192.168.50.2:554/Streaming/Channels/102" -t 10 test.mp4
```

| Pemeriksaan | Catatan |
|---|---|
| Nama pengguna / kata sandi | Karakter khusus (`@`, `:`, `/`, `#`) harus di-URL-encode (`@` → `%40`) |
| Channel | `101` main stream, `102` sub stream (dipakai proyek ini) |
| Port 554 aktif di pengaturan kamera | |
| Jumlah sambungan | Tutup VLC atau program lain yang membuka stream yang sama |
| Firewall | `sudo ufw status`; izinkan RTSP bila perlu |

Format URL merek lain (Dahua, generik) ada di `troubleshooting.md` wise78.

### Problem: latensi tinggi, frame hilang, gambar hijau / artefak

**Penyebab:** paket UDP hilang, jaringan padat, atau beban pemrosesan membuat buffer menumpuk.

```bash
# Paksa RTSP lewat TCP (backend FFmpeg OpenCV)
export OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"
python3 main_app.py
```

- Pakai kabel LAN/PoE, bukan Wi-Fi.
- Pakai sub stream (`102`) dan turunkan *frame rate* / *bitrate* di pengaturan kamera.
- Opsi `use_gstreamer`, `read_width`, `read_height` pada `camera_config.json` milik wise78 hanya berlaku untuk `data_collector.py`, bukan `main_app.py`.
- Pastikan hanya frame terbaru yang diproses (jangan menumpuk antrean frame).

### Problem: kamera tidak ditemukan di jaringan

- Pakai SADP atau HiTools Hikvision.
- Pastikan kamera dan Pi satu subnet (mis. `192.168.50.x/24`).
- Bila PoE, periksa switch PoE dan kabel.

### Problem: kata sandi kamera tampil di kode / terunggah

Hapus dari riwayat, ganti kata sandi kamera, dan pakai variabel lingkungan:

```python
import os
PASSWORD = os.environ.get("CAMERA_PASSWORD", "")
```

### Problem: kualitas gambar buruk atau tidak ada citra tersimpan saat pengumpulan dataset

Terkait `data_collector.py` (filter ketajaman, kecerahan, anti-duplikat): lihat *Data Collection Issues* pada `troubleshooting.md` wise78. Perbaiki juga pencahayaan, eksposur kamera, dan kebersihan lensa.

---

## 6. 🌐 Jaringan dan Akses Dashboard

### Problem: dashboard tidak terbuka dari perangkat lain

| Pemeriksaan | Solusi |
|---|---|
| Satu jaringan dengan Pi? | Hubungkan ke Wi-Fi/LAN yang sama |
| Alamat benar? | `http://<IP-PI>:5000` (bukan `https`) |
| Flask mendengarkan semua antarmuka? | `app.run(host="0.0.0.0", port=5000)` |
| Program berjalan? | `ss -tlnp \| grep 5000` |
| Firewall? | `sudo ufw status`; buka port 5000 |
| Port bentrok? | `sudo lsof -i :5000` lalu hentikan prosesnya |

### Mengetahui IP Raspberry Pi

```bash
hostname -I
ip -4 addr show
```

### Mengganti jaringan Wi-Fi Raspberry Pi OS

```bash
sudo nmtui
# atau
nmcli device wifi list
sudo nmcli device wifi connect "NAMA_WIFI" password "KATA_SANDI"
```

### Melihat IP perangkat yang terhubung ke hotspot iPhone

Hotspot iPhone tidak menampilkan daftar perangkat. Dari Pi:

```bash
hostname -I
ip route | grep default
sudo apt install -y nmap
sudo nmap -sn 172.20.10.0/28        # subnet hotspot iPhone biasanya 172.20.10.x
```

### Problem: Pi dan kamera tidak saling terhubung di hotspot ponsel

Hotspot ponsel sering mengisolasi klien dan tidak cocok dengan kamera IP bersubnet tetap. Pakai router/switch lokal, atau samakan subnet kamera dengan jaringan yang dipakai.

---

## 7. 🎯 Deteksi dan Tracking

### Problem: satu kerusakan terhitung berkali-kali (N membengkak)

Tracker gagal menganggapnya objek yang sama.

| Gejala | Penyesuaian di `main_app.py` |
|---|---|
| Objek yang sama dihitung ganda | Naikkan `WAKTU_HILANG_MAX_DETIK`, turunkan `IOU_MIN_TRACKING`, naikkan `KECEPATAN_TRACKING_PIKSEL_PER_DETIK` |
| Dua kerusakan berbeda digabung | Naikkan `IOU_MIN_TRACKING`, turunkan `TOLERANSI_JARAK_DASAR` |

Nilai saat ini: IoU ≥ 0,2; jarak 20 + 150·Δt piksel; waktu hilang 1 s. Tracker di-reset tiap sesi.

### Problem: kerusakan nyata tidak terdeteksi (sesi `N = 0`)

- Periksa pencahayaan dan sudut kamera.
- Pastikan ROI menutupi area belt.
- Turunkan `CONF_THRESHOLD`.
- Naikkan *shutter* kamera bila gambar buram oleh gerak belt.
- Satu sesi hanya mencakup segmen belt yang terlihat; kerusakan di luar jendela memang tidak terdeteksi pada sesi itu.

### Problem: kerusakan terdeteksi sesaat lalu hilang

Normal bila objek melewati ROI cepat. Perpanjang `DURASI_SESI_DETIK` atau perlebar ROI.

### Problem: hasil berbeda dengan penilaian teknisi

Periksa dahulu apakah label teknisi **per sesi** (segmen yang terlihat), bukan per seluruh belt. Untuk keputusan seluruh belt, gunakan Mode Custom Sesi.

---

## 8. 📊 Sesi, Mode Custom Sesi, dan Dashboard

### Problem: tombol **Mulai** tidak aktif / dashboard terkunci setelah Stop

Versi lama tidak mengirim penutup batch saat **Stop**. Versi terbaru `main_app.py` sudah memperbaikinya. Pastikan `main_app.py` dan `templates/dashboard.html` sama-sama versi terbaru, lalu muat ulang halaman (Ctrl + F5).

### Problem: dashboard tidak berubah setelah `dashboard.html` diganti

- Letakkan di folder `templates/`.
- Muat ulang paksa (Ctrl + F5).
- Pastikan Flask memakai berkas yang benar, bukan salinan lama di folder lain.

### Problem: keputusan akhir Mode Custom Sesi tidak muncul

- Pastikan semua sesi selesai atau **Stop** ditekan.
- Keputusan akhir mengikuti sesi dengan keparahan tertinggi: Tidak Rusak < Rusak Ringan < Rusak Berat. Bila seri, dipilih skor S tertinggi.

### Problem: video tidak tampil

- Buka `/video_feed` langsung di browser.
- Frame sementara memakai `/dev/shm`; periksa `df -h /dev/shm`.
- Pastikan kedua proses (deteksi dan server) berjalan.

### Problem: data real-time (SSE) berhenti

- Buka `/stream` di browser.
- Muat ulang halaman; mode hemat daya browser atau proxy dapat memutus SSE.
- Lihat pesan galat di terminal.

### Problem: durasi sesi tidak sesuai harapan

`DURASI_SESI_DETIK` dibaca saat program dimulai. Ubah, simpan, jalankan ulang. Durasi harus sama untuk data latih, data uji, dan operasi.

---

## 9. 🗄️ Database

### Problem: `sqlite3.OperationalError: database is locked`

- Pastikan tidak ada dua program menulis bersamaan.
- Database memakai mode WAL; berkas `-wal` dan `-shm` adalah normal.
- Tutup alat lain (mis. DB Browser) yang membuka database dalam mode tulis.

### Problem: menyalin database dari Pi

Hentikan `main_app.py`, lalu salin `conveyor_monitoring.db` beserta `-wal` dan `-shm` bila ada.

### Problem: `category_logreg` / `prob_berat_logreg` berisi NULL

| Penyebab | Penjelasan |
|---|---|
| Model `.pkl` belum ada saat program dimulai | Data latih memang kosong; normal |
| Sesi `N = 0` | Model tidak dijalankan untuk sesi tanpa deteksi; `bandingkan_metode.py` menganggapnya "Tidak Rusak" |
| Jalur model salah | Periksa `LOGREG_MODEL_PATH` |

### Problem: `r_max`, `c_avg`, `score_s` NULL atau 0 pada sesi `N = 0`

Sesi tanpa deteksi tidak memiliki nilai yang terdefinisi. Pastikan format konsisten (NULL pada `db_severity.py`, 0 pada data sintetis) saat agregasi dan filter. Filter pelatihan sebaiknya memakai `n_detections >= 1`.

### Problem: label teknisi atau catatan hilang

`update_label_teknisi()` menimpa kolom `catatan`. Berikan label dan catatan dalam satu pemanggilan.

### Problem: timestamp tidak rapi (data sintetis)

Awal sesi mengikuti kelipatan `durasi + jeda`. Atur `--durasi` dan `--jeda` pada `generate_synthetic_db.py`.

---

## 10. 📈 Kalibrasi, Regresi Logistik, dan Perbandingan Metode

### Problem: data terlalu sedikit untuk kalibrasi

Dibutuhkan minimal 30 sesi berlabel (ideal 60-80 atau lebih), dengan **kedua kelas** (Ringan dan Berat) cukup terwakili. Lengkapi label lewat `update_label_teknisi()`.

### Problem: parameter di `main_app.py` tidak sama dengan hasil kalibrasi

Hasil kalibrasi tidak otomatis masuk. Salin manual `W_R, W_N, W_C, TAU, N_MAX`. Bila `N_MAX` berubah, **latih ulang** regresi logistik (karena `N_norm` ikut berubah), lalu rekam data uji baru.

### Problem: `severity_logreg.pkl` tidak ditemukan / tidak terpakai

- Letakkan di folder yang sama dengan `main_app.py` atau ubah `LOGREG_MODEL_PATH`.
- Model dimuat **sekali saat program dimulai**; jalankan ulang setelah meletakkan model.
- Versi `scikit-learn` saat melatih dan memakai harus sama (`InconsistentVersionWarning` menandakan ketidakcocokan).

### Problem: precision/recall/F1 rule-based terlihat sempurna (1,000)

Pada data kecil dengan kelas yang terpisah jelas, ini mungkin terjadi. Periksa jumlah data uji (di bawah ~30 sesi rapuh), apakah parameter pernah disetel dengan melihat data uji (kebocoran data), dan apakah data latih dan uji tercampur. Laporkan jumlah sesi dan interval kepercayaan Kappa.

### Problem: Kappa regresi logistik lebih rendah dari rule-based

Dapat terjadi dan sah: rule-based yang dikalibrasi langsung pada fitur yang sama sering sama baik atau lebih baik pada data kecil. Jangan mengubah data atau label demi membalik hasil; jelaskan sebagai temuan di laporan.

### Problem: Kappa rendah padahal akurasi tinggi

Kelas tidak seimbang atau banyak sesi dekat ambang. Periksa confusion matrix dan sesi yang salah klasifikasi satu per satu.

### Problem: hasil 3 kelas dan biner berbeda

Skenario A (3 kelas) memasukkan sesi `N = 0`; skenario B (biner) hanya sesi Ringan/Berat dengan deteksi. Sebutkan skenario yang dipakai.

### Problem: memisahkan data latih dan data uji pada database yang sama

Pakai batas waktu, atau simpan data uji di database terpisah:

```bash
python3 bandingkan_metode.py --db conveyor_monitoring.db --sejak "2026-10-20 00:00:00"
```

### Problem: Kappa tidak dapat dihitung / hanya satu kelas

Data uji memuat satu kelas saja. Tambah sesi dari kondisi lain.

---

## 11. 🌡️ Performa dan Suhu

### Problem: FPS rendah / peringatan suhu

```bash
vcgencmd measure_temp
vcgencmd get_throttled        # 0x0 = normal
htop
free -h
```

| Gejala | Solusi |
|---|---|
| Suhu > 70 °C | Pasang heatsink/kipas, perbaiki sirkulasi udara |
| `get_throttled` ≠ `0x0` | Catu daya kurang atau suhu terlalu tinggi |
| FPS rendah | Pastikan model berjalan di Edge TPU; pakai sub stream; persempit ROI; perkecil `IMGSZ` bila memungkinkan (perlu konversi ulang) |
| Coral panas | Beri ventilasi; tetap pakai `libedgetpu1-std` |

Optimasi sistem lain (menonaktifkan layanan yang tidak perlu, *swap*) dibahas di `troubleshooting.md` wise78. Catatan: pengaturan *GPU memory split* dan *overclock* pada panduan itu merujuk Pi 4; di Pi 5 periksa ulang kebutuhannya sebelum mengubah.

### Problem: memori tinggi

```bash
free -h
ps aux --sort=-%mem | head -10
```

Tutup proses yang tidak perlu; tambah *swap* bila perlu (langkah ada di `troubleshooting.md` wise78).

### Problem: kartu SD cepat aus atau penuh

- Frame sementara memakai `/dev/shm` (RAM), bukan SD.
- Pantau ruang dengan `df -h`.
- Pindahkan atau cadangkan database berkala.

### Problem: program berhenti sendiri setelah lama berjalan

Lihat terminal/log. Penyebab umum: suhu, daya, RTSP putus, memori penuh. Jalankan sebagai layanan `systemd` agar hidup kembali otomatis.

---

## 12. 🖥️ Simulator dan Data Sintetis

### Problem: simulator tidak menampilkan deteksi pada percepatan tinggi

Kerusakan dapat terlewat antar langkah waktu bila percepatan terlalu besar. Versi terbaru sudah menangani ini; turunkan `--kecepatan` bila masih terlewat.

### Problem: `TemplateNotFound: dashboard.html`

Buat folder `templates/` di samping skrip dan letakkan `dashboard.html` di dalamnya.

### Problem: port 5000 dipakai

```bash
python3 simulator_dashboard.py --port 5001
```

### Problem: menghentikan simulator

Tekan `Ctrl + C`. Jangan memakai `pkill -f` dengan pola yang cocok dengan perintah shell sendiri; pakai `ps aux | grep '[s]imulator_dashboard'` lalu `kill <PID>`.

### Peringatan: data sintetis bukan hasil penelitian

Database sintetis hanya untuk menguji program analisis. Hasil BAB 4 harus berasal dari data lapangan nyata.

---

## 13. 🔧 Advanced Troubleshooting

### Debug

```bash
python3 main_app.py 2>&1 | tee run.log          # simpan keluaran terminal
sudo journalctl -u conveyor-detection -f        # bila dijalankan sebagai layanan systemd
```

### Profiling CPU

```bash
python3 -m cProfile -o profile.stats main_app.py
python3 -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(10)"
```

### Diagnostik jaringan RTSP

```bash
ffprobe -v quiet -print_format json -show_format -show_streams "rtsp://..."
ping -c 100 192.168.50.2 | grep "packet loss"
```

---

## 14. 📡 Monitoring dan Pemulihan

### Skrip pemantauan sederhana

```bash
#!/bin/bash
# monitor_system.sh
TEMP=$(vcgencmd measure_temp | cut -d'=' -f2 | cut -d"'" -f1)
MEMORY=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
DISK=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')

(( $(echo "$TEMP > 70" | bc -l) ))   && echo "ALERT: suhu tinggi: ${TEMP}°C"
(( $(echo "$MEMORY > 85" | bc -l) )) && echo "ALERT: memori tinggi: ${MEMORY}%"
[ "$DISK" -gt 90 ]                   && echo "ALERT: disk hampir penuh: ${DISK}%"
```

### Pemulihan otomatis (opsional)

```bash
#!/bin/bash
# auto_recovery.sh
if ! pgrep -f "main_app.py" > /dev/null; then
    echo "Memulai ulang main_app.py..."
    cd /home/pi/conveyor_project && source ~/conveyor_env/bin/activate
    nohup python3 main_app.py >> run.log 2>&1 &
fi
```

> Tidak disarankan menghapus log otomatis atau me-*reboot* sistem saat pengambilan data berlangsung karena dapat memutus sesi yang sedang direkam.

---

## 15. 📞 Getting Help

### Sebelum meminta bantuan

1. **Kumpulkan informasi**

   ```bash
   uname -a
   cat /etc/os-release
   vcgencmd measure_temp; vcgencmd get_throttled
   lsusb
   python --version
   pip list | grep -i -E "pycoral|tflite|opencv|flask|scikit|numpy"
   ```

2. **Periksa log**: keluaran terminal `main_app.py` dan `tail -100 /var/log/syslog`.
3. **Uji komponen satu per satu**: kamera (`ffmpeg`), Coral (`edgetpu.list_edge_tpus()`), model (muat `Interpreter`), dashboard (`curl`).

### Template laporan masalah

1. **Informasi sistem**: model Raspberry Pi, versi OS, versi Python, paket terpasang.
2. **Detail galat**: pesan lengkap, langkah mereproduksi, perilaku yang diharapkan vs aktual.
3. **Log**: keluaran terminal dan log sistem.
4. **Konfigurasi**: konstanta di `main_app.py` (tanpa kata sandi), versi model, `DURASI_SESI_DETIK`.

### Rujukan

- Repository wise78: <https://github.com/wise78/Conveyor-Belt-Damage-Detection> (Issues dan `docs/`)
- Panduan instalasi dan penggunaan proyek ini: [`GUIDEBOOK.md`](GUIDEBOOK.md)

---

## 16. ✅ Checklist Sebelum Pengambilan Data

- [ ] Coral terdeteksi dan suhu Pi normal
- [ ] RTSP stabil (uji beberapa menit tanpa artefak)
- [ ] ROI sudah ditetapkan dan menutupi area belt
- [ ] Model `.tflite` benar (`_edgetpu`, INT8, masukan 320)
- [ ] `DURASI_SESI_DETIK` sudah final dan sama untuk semua tahap
- [ ] Pencahayaan dan sudut kamera tetap
- [ ] Kata sandi kamera tidak ada di berkas yang dibagikan
- [ ] Kriteria label teknisi (Ringan/Berat) tertulis; label per sesi
- [ ] Data latih dan data uji dapat dibedakan (timestamp atau database terpisah)
- [ ] Parameter rule-based dan model regresi logistik final **sebelum** data uji direkam
- [ ] Cadangan database disimpan di luar Pi

---

**Catatan:** cadangkan konfigurasi dan data sebelum mengubah apa pun, dan uji solusi di lingkungan aman terlebih dahulu.
