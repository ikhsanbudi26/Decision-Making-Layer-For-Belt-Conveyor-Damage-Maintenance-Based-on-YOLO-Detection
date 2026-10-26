#!/usr/bin/env python3
"""
simulator_dashboard.py - Simulator dashboard TANPA Raspberry Pi, kamera, atau EdgeTPU.

Menjalankan dashboard.html yang SAMA dengan sistem asli, dengan "belt virtual":
video palsu (animasi belt + kotak deteksi), deteksi acak yang konsisten dengan
satu putaran belt, dan logika sesi/batch/keputusan akhir (skenario terburuk)
yang sama dengan main_app.py. Tidak memakai OpenCV, YOLO, atau database.

Kebutuhan:  pip install flask pillow
Struktur:   simulator_dashboard.py
            templates/dashboard.html      <- dashboard.html milikmu
Jalankan:   python3 simulator_dashboard.py
Buka:       http://localhost:5000

Contoh:
  python3 simulator_dashboard.py --kondisi berat            # belt rusak berat
  python3 simulator_dashboard.py --kondisi bersih --durasi 30
  python3 simulator_dashboard.py --kecepatan 1               # waktu nyata (tanpa percepatan)

CATATAN: simulator ini hanya untuk mencoba tampilan dan alur sistem. Angkanya
BUKAN data penelitian.
"""
import argparse, io, json, math, os, queue, random, threading, time

from flask import Flask, Response, render_template, request
from PIL import Image, ImageDraw, ImageFont

# ---- Salinan konfigurasi dari main_app.py (sesuaikan bila parameter final berubah) ----
W_R, W_N, W_C = 0.4, 0.3, 0.3
TAU = 0.4636
N_MAX = 5
ROI_AWAL = [100, 100, 900, 600]
FRAME_W, FRAME_H = 1280, 720
METODE_UTAMA = "rule_based"   # simulator hanya menghitung rule-based

# ---- Salinan logika keputusan akhir dari main_app.py ----
URUTAN_KEPARAHAN = {"Tidak Rusak": 0, "Rusak Ringan": 1, "Rusak Berat": 2}


def ringkas_batch(hasil_sesi):
    penentu = max(hasil_sesi, key=lambda h: (URUTAN_KEPARAHAN[h["kategori"]], h["score_s"] or 0.0))
    rincian = {k: sum(1 for h in hasil_sesi if h["kategori"] == k) for k in URUTAN_KEPARAHAN}
    return {
        "kategori_akhir": penentu["kategori"], "jumlah_sesi": len(hasil_sesi),
        "rincian": rincian, "sesi_penentu": penentu["session_id"],
        "n_detections": penentu["n_detections"], "r_max": penentu["r_max"],
        "c_avg": penentu["c_avg"], "score_s": penentu["score_s"],
    }


def klasifikasi_rule_based(r_max, n_norm, c_avg):
    score = W_R * r_max + W_N * n_norm + W_C * c_avg
    return score, ("Rusak Berat" if score > TAU else "Rusak Ringan")


# =======================================================
# BELT VIRTUAL: kerusakan tetap di posisi tertentu sepanjang satu putaran
# =======================================================
CLASSES = ["Tear", "Hole", "Puncture", "Patch Work"]
DURASI_TAMPIL = 2.5   # detik-simulasi sebuah kerusakan terlihat melintasi ROI
P_DETEKSI = 0.92      # peluang model mendeteksi kerusakan yang terlihat


def buat_belt(kondisi, putaran, rng):
    """Daftar kerusakan tetap pada belt: (waktu posisi dalam putaran, kelas, rasio luas, conf)."""
    if kondisi == "bersih":
        n_kecil, n_besar = 0, 0
    elif kondisi == "ringan":
        n_kecil, n_besar = rng.randint(2, 3), 0
    elif kondisi == "berat":
        n_kecil, n_besar = rng.randint(2, 3), rng.randint(2, 3)
    else:  # campuran
        n_kecil, n_besar = rng.randint(2, 4), 1
    cacat = []
    for i in range(n_kecil + n_besar):
        besar = i >= n_kecil
        r = rng.uniform(0.18, 0.40) if besar else rng.uniform(0.02, 0.12)
        conf = rng.uniform(0.65, 0.92) if besar else rng.uniform(0.45, 0.78)
        cacat.append({"t": rng.uniform(0, putaran), "kelas": rng.choice(CLASSES),
                      "r": r, "conf": conf, "y": rng.uniform(0.15, 0.85)})
    return cacat


# =======================================================
# SIMULATOR (menggantikan detection_process)
# =======================================================
class Simulator:
    FPS = 10

    def __init__(self, args):
        self.a = args
        self.rng = random.Random(args.seed)
        self.cacat = buat_belt(args.kondisi, args.putaran, self.rng)
        self.cmd = queue.Queue()
        self.sim_t = 0.0               # jam simulasi (detik); belt terus berputar
        self.roi = list(ROI_AWAL)
        self.frame_bytes = b""
        self.lock = threading.Lock()
        self.clients, self.clients_lock = [], threading.Lock()
        self.font = self._font(22)
        self.last_real = time.time()
        self.info = "menunggu"
        self.terlihat = []             # kerusakan yang sedang tampak (untuk digambar)
        self.nomor_sesi = 0
        self.fase_prev, self.dt_sim = 0.0, 0.0

    @staticmethod
    def _font(size):
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

    # ---------- SSE ----------
    def emit(self, d):
        payload = json.dumps(d)
        with self.clients_lock:
            for q in list(self.clients):
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    self.clients.remove(q)

    # ---------- waktu ----------
    def tick(self):
        now = time.time()
        self.fase_prev = self.fase_belt()
        self.dt_sim = (now - self.last_real) * self.a.kecepatan
        self.sim_t += self.dt_sim
        self.last_real = now

    def fase_belt(self):
        return self.sim_t % self.a.putaran

    # ---------- render frame palsu ----------
    def kotak_cacat(self, c, progres):
        x1, y1, x2, y2 = self.roi
        rw, rh = x2 - x1, y2 - y1
        area = c["r"] * rw * rh
        h = max(20, min(rh * 0.9, math.sqrt(area / 1.6)))
        w = max(20, min(rw * 0.9, 1.6 * h))
        x = x2 - w - progres * (rw - w)     # bergerak dari kanan ke kiri ROI
        y = y1 + c["y"] * (rh - h)
        return int(x), int(y), int(x + w), int(y + h)

    def render(self, sesi_aktif):
        img = Image.new("RGB", (FRAME_W, FRAME_H), (38, 40, 46))
        d = ImageDraw.Draw(img)
        # belt: pita gelap dengan garis melintang yang bergeser
        d.rectangle([0, 60, FRAME_W, FRAME_H - 60], fill=(72, 72, 78))
        geser = int((self.sim_t * 150) % 80)
        for gx in range(-80, FRAME_W + 80, 80):
            d.line([(gx - geser + 80, 60), (gx - geser + 80, FRAME_H - 60)], fill=(60, 60, 66), width=3)
        x1, y1, x2, y2 = self.roi
        warna = (0, 255, 255)
        d.rectangle([x1, y1, x2, y2], outline=warna, width=3)
        d.text((x1, max(y1 - 28, 4)), "ROI" if sesi_aktif else "ROI (menunggu sesi)", fill=warna, font=self.font)
        for kotak, teks in self.terlihat:
            d.rectangle(list(kotak), outline=(0, 255, 0), width=3)
            d.text((kotak[0], max(kotak[1] - 26, 4)), teks, fill=(0, 255, 0), font=self.font)
        d.text((12, 8), f"SIMULATOR  |  fase belt {self.fase_belt():5.1f}/{self.a.putaran:.0f} dtk  |  {self.info}",
               fill=(255, 210, 90), font=self.font)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        with self.lock:
            self.frame_bytes = buf.getvalue()

    def langkah(self, sesi_aktif, sudah=None, dets=None):
        """Satu langkah waktu: majukan jam, cari kerusakan yang tampak, gambar frame.
        Kerusakan yang sedang terlihat saat sesi dimulai tetap dihitung sebagai
        deteksi baru (tracker di-reset tiap sesi, seperti sistem asli)."""
        self.tick()
        fase = self.fase_belt()
        self.terlihat = []
        for i, c in enumerate(self.cacat):
            sejak = (fase - c["t"]) % self.a.putaran
            # kerusakan dianggap terlihat bila sedang tampak ATAU awal kemunculannya
            # terlewati dalam langkah waktu ini (penting saat percepatan tinggi)
            terlewati = ((c["t"] - self.fase_prev) % self.a.putaran) <= self.dt_sim
            if sejak < DURASI_TAMPIL or terlewati:
                sejak = min(sejak, DURASI_TAMPIL)
                kotak = self.kotak_cacat(c, sejak / DURASI_TAMPIL)
                dilihat_model = sudah is not None and (i in sudah)
                if sesi_aktif and sudah is not None and i not in sudah:
                    if self.rng.random() < P_DETEKSI:
                        sudah.add(i)
                        dets.append(c)
                        self.emit({"type": "deteksi_baru", "class_name": c["kelas"],
                                   "confidence": c["conf"], "area_ratio": c["r"]})
                        dilihat_model = True
                    else:
                        sudah.add(i)   # terlewat model pada lewatan ini
                if dilihat_model or not sesi_aktif:
                    self.terlihat.append((kotak, f"{c['kelas']}: {c['conf']:.2f}"))
        self.render(sesi_aktif)

    # ---------- perintah ----------
    def cek_perintah(self, st):
        while True:
            try:
                cmd = self.cmd.get_nowait()
            except queue.Empty:
                break
            t = cmd.get("cmd")
            if t == "start_session":
                st["session_trigger"] = True
            elif t == "start_batch":
                st.update(batch_mode=True, batch_aktif=True, batch_sisa=cmd.get("jumlah_sesi"),
                          batch_target=cmd.get("jumlah_sesi"), batch_hasil=[], session_trigger=True)
            elif t == "stop_batch":
                st["batch_mode"] = False
            elif t == "set_roi":
                self.roi = cmd["coords"]

    # ---------- loop utama (struktur sama dengan detection_process) ----------
    def run(self):
        st = dict(session_trigger=False, batch_mode=False, batch_aktif=False,
                  batch_sisa=None, batch_target=None, batch_hasil=[])
        self.emit({"type": "menunggu"})
        print("[SIM] Siap. Menunggu perintah dari dashboard...")
        while True:
            self.cek_perintah(st)
            if not st["session_trigger"] and not st["batch_mode"]:
                self.info = "menunggu"
                self.langkah(False)
                time.sleep(1 / self.FPS)
                continue

            st["session_trigger"] = False
            self.nomor_sesi += 1
            session_id = f"S-SIM{self.nomor_sesi:04d}"
            self.emit({"type": "sesi_mulai", "session_id": session_id, "lokasi": None})
            print(f"[SIM] Mulai {session_id} (durasi {self.a.durasi:g} dtk-simulasi)")
            t_mulai = self.sim_t
            sudah, dets = set(), []
            while self.sim_t - t_mulai < self.a.durasi:
                self.cek_perintah(st)
                self.info = f"{session_id} {self.sim_t - t_mulai:4.1f}/{self.a.durasi:g} dtk"
                self.langkah(True, sudah, dets)
                time.sleep(1 / self.FPS)

            n = len(dets)
            if n == 0:
                kategori, score, r_max, c_avg = "Tidak Rusak", None, None, None
            else:
                r_max = max(d["r"] for d in dets)
                c_avg = sum(d["conf"] for d in dets) / n
                score, kategori = klasifikasi_rule_based(r_max, min(n / N_MAX, 1.0), c_avg)
            self.emit({"type": "sesi_selesai", "session_id": session_id, "kategori": kategori,
                       "n_detections": n, "r_max": r_max, "c_avg": c_avg, "score_s": score,
                       "kategori_logreg": None, "prob_logreg": None})
            print(f"[SIM] Selesai {session_id}: {kategori} (n={n})")

            if st["batch_aktif"]:
                st["batch_hasil"].append({"session_id": session_id, "kategori": kategori,
                                          "n_detections": n, "r_max": r_max,
                                          "c_avg": c_avg, "score_s": score})
                if st["batch_sisa"] is not None:
                    st["batch_sisa"] -= 1
                selesai = (not st["batch_mode"]) or (st["batch_sisa"] is not None and st["batch_sisa"] <= 0)
                if selesai:
                    st["batch_mode"] = st["batch_aktif"] = False
                    ring = ringkas_batch(st["batch_hasil"])
                    print(f"[SIM] KEPUTUSAN AKHIR: {ring['kategori_akhir']} {ring['rincian']}")
                    self.emit({"type": "batch_selesai", **ring})
                else:
                    self.emit({"type": "batch_progress", "selesai": len(st["batch_hasil"]),
                               "target": st["batch_target"]})
                    st["session_trigger"] = True
                    # jeda transisi singkat antar sesi (mirip pengurasan frame di sistem asli)
                    self.sim_t += self.a.jeda
                    continue
            self.emit({"type": "menunggu"})


# =======================================================
# FLASK -- endpoint sama dengan main_app.py
# =======================================================
def buat_app(sim):
    base = os.path.dirname(os.path.abspath(__file__))
    tpl = os.path.join(base, "templates")
    app = Flask(__name__, template_folder=tpl if os.path.isdir(tpl) else base)

    @app.route("/")
    def index():
        return render_template("dashboard.html")

    @app.route("/start_session", methods=["POST"])
    def start_session():
        sim.cmd.put({"cmd": "start_session"}); return {"status": "ok"}, 200

    @app.route("/start_batch", methods=["POST"])
    def start_batch():
        data = request.get_json(silent=True) or {}
        j = data.get("jumlah_sesi")
        sim.cmd.put({"cmd": "start_batch", "jumlah_sesi": int(j) if j else None})
        return {"status": "ok"}, 200

    @app.route("/stop_batch", methods=["POST"])
    def stop_batch():
        sim.cmd.put({"cmd": "stop_batch"}); return {"status": "ok"}, 200

    @app.route("/set_roi", methods=["POST"])
    def set_roi():
        d = request.get_json(silent=True) or {}
        try:
            x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
        except (KeyError, TypeError, ValueError):
            return {"status": "error", "message": "Koordinat ROI tidak valid"}, 400
        if x2 <= x1 or y2 <= y1:
            return {"status": "error", "message": "ROI harus punya lebar & tinggi positif"}, 400
        sim.cmd.put({"cmd": "set_roi", "coords": [x1, y1, x2, y2]})
        return {"status": "ok", "roi": [x1, y1, x2, y2]}, 200

    @app.route("/video_feed")
    def video_feed():
        def gen():
            while True:
                with sim.lock:
                    fb = sim.frame_bytes
                if fb:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + fb + b"\r\n"
                time.sleep(0.1)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/stream")
    def stream():
        def gen():
            q = queue.Queue(maxsize=100)
            with sim.clients_lock:
                sim.clients.append(q)
            try:
                while True:
                    yield f"data: {q.get()}\n\n"
            finally:
                with sim.clients_lock:
                    if q in sim.clients:
                        sim.clients.remove(q)
        return Response(gen(), mimetype="text/event-stream")

    return app


def main():
    ap = argparse.ArgumentParser(description="Simulator dashboard conveyor belt (tanpa Raspberry)")
    ap.add_argument("--kondisi", choices=["bersih", "ringan", "berat", "campuran"], default="campuran",
                    help="kondisi belt virtual (bawaan: campuran)")
    ap.add_argument("--durasi", type=float, default=30, help="durasi satu sesi, detik-simulasi (bawaan 30)")
    ap.add_argument("--putaran", type=float, default=300, help="waktu satu putaran belt, detik-simulasi (bawaan 300)")
    ap.add_argument("--kecepatan", type=float, default=10, help="percepatan waktu (bawaan 10x; 1 = waktu nyata)")
    ap.add_argument("--jeda", type=float, default=1.0, help="jeda transisi antar sesi, detik-simulasi")
    ap.add_argument("--seed", type=int, default=7, help="seed posisi kerusakan di belt")
    ap.add_argument("--port", type=int, default=5000)
    a = ap.parse_args()

    sim = Simulator(a)
    print(f"[SIM] Belt virtual '{a.kondisi}': {len(sim.cacat)} kerusakan di sepanjang 1 putaran ({a.putaran:g} dtk). "
          f"Sesi {a.durasi:g} dtk, percepatan {a.kecepatan:g}x -> 1 sesi ~ {a.durasi / a.kecepatan:.1f} dtk nyata.")
    threading.Thread(target=sim.run, daemon=True).start()
    buat_app(sim).run(host="0.0.0.0", port=a.port, threaded=True)


if __name__ == "__main__":
    main()
