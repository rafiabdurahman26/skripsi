# Drone Human Detection

Deteksi manusia (YOLOv8n, ONNX) pada live stream **DJI Tello**, dengan deteksi murni
**onnxruntime** — tanpa torch/ultralytics. GPU (CUDA) otomatis dipakai di laptop
ber-NVIDIA, fallback CPU di device lain. Sumber video bisa drone Tello atau
kamera HP (IP Webcam) sebagai cadangan.

## Cara Pakai

Setup sekali (dijalankan dari terminal, bukan double-click — biar error terlihat):

```
setup.bat
```

Setup otomatis:
- Deteksi GPU NVIDIA (`nvidia-smi`) → install dependensi GPU
- Tidak ada NVIDIA → install versi CPU
- Prioritas `uv` bila terpasang, fallback `pip` + venv

Jalankan:

```
uv run python main.py --source tello   # drone Tello (default)
uv run python main.py --source phone   # kamera HP (IP Webcam)
```

Tanpa uv, pakai `.venv\Scripts\python.exe main.py ...`.

### Opsi

| Opsi | Nilai | Default | Keterangan |
|---|---|---|---|
| `--source` | `tello` / `phone` | `tello` | Sumber video |
| `--model` | path `.onnx` | `best_model_seed42.onnx` | Model deteksi |
| `--device` | `auto` / `cuda` / `cpu` | `auto` | Paksa provider |
| `--conf` | 0.0–1.0 | `0.25` | Ambang confidence |
| `--phone-url` | URL | lihat `config.py` | URL MJPEG IP Webcam |

Selama live stream: `q`/`Q` quit, `t` takeoff/land. Overlay deteksi (kotak + jumlah
orang + FPS) digambar langsung di HUD Tello.

## Instalasi Manual

```
uv sync --extra gpu        # NVIDIA / CUDA
uv sync --extra cpu        # tanpa NVIDIA
```

Pip:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-gpu.txt   # atau requirements.txt untuk CPU
```

## Catatan Teknis

- `detector.py` — inference onnxruntime murni: auto-detect CUDA → CPU, letterbox 640,
  NMS numpy, self-check `python detector.py`. Provider CUDA memakai
  `cudnn_conv_algo_search=EXHAUSTIVE` (≈3x lebih cepat dari default) + buffer
  blob/letterbox yang di-reuse — hasil: **±82 FPS end-to-end** di RTX 2050.
  Provider aktif terlihat live di HUD (`PERSON n | FPS xx | CUDA`).
- GPU runtime (CUDA 13 + cuDNN 9) ikut ter-install **di dalam venv** via
  `onnxruntime-gpu[cuda,cudnn]`; toolkit CUDA 12.1 sistem dan project lain tidak
  disentuh. `nvidia-cublas` di-pin `13.6.0.2` (versi terbaru tidak punya wheel
  Windows).
- Kode Tello (`drone.py`, `input_handler.py`, `video_handler.py`, `config.py`)
  diambil utuh dari project `dji-tello` — source asli tidak diubah.
- Folder `captures/` (hasil rekam) dan `config.json` tidak di-commit.

## Laporan Sesi (Reports)

Setiap run otomatis mencatat dan menulis laporan (mekanisme mengikuti project
skripsi `drone_e99_face_recognition`: CSV per-sesi + manifest JSON sebagai
single source, Excel selalu di-rebuild dari source):

```
reports/
  inference/session_<ts>/detection_log_<ts>.csv      # per deteksi (ts, conf, bbox, fps)
                        per_second_<ts>.csv           # agregasi per detik
                        session_stats_<ts>.csv        # ringkasan sesi
                        persons_per_second_<ts>.png   # grafik (cv2, tanpa matplotlib)
  runs/<ts>_detection.json                            # manifest sesi
  summary/summary_detection.xlsx                      # SessionLog + PerSecond + PlotsGrid
```

Grafik digambar dengan OpenCV (tanpa matplotlib); satu-satunya dependensi
tambahan `openpyxl`. XLSX selalu dibangun ulang dari manifest+CSV, tidak pernah
di-append — file lama tidak perlu di-reset. Self-check: `uv run python reporting.py`.
