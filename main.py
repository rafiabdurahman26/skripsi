"""
Deteksi manusia real-time dari kamera HP menggunakan model YOLOv8n (ONNX).
Jalankan: python realtime_detect.py
Tekan 'q' untuk keluar.
"""

import cv2
import time
from ultralytics import YOLO

# ── Konfigurasi ──────────────────────────────────────────
MODEL_PATH = r"D:\punya aa\KULIAH\skripsi\best_model_seed42.onnx"

# Ganti dengan URL dari app IP Webcam di HP Anda
PHONE_CAM_URL = "http://192.168.251.201:8080/video"

DEVICE = "cpu"          # ← diubah dari "cuda:0" ke "cpu"
CONF_THRESH = 0.25      # ambang kepercayaan deteksi


def main():
    print("Memuat model...")
    model = YOLO(MODEL_PATH, task="detect")
    print(f"✅ Model dimuat, dijalankan di device: {DEVICE}")

    cap = cv2.VideoCapture(PHONE_CAM_URL)
    if not cap.isOpened():
        print(f"❌ Tidak bisa membuka stream kamera: {PHONE_CAM_URL}")
        print("   Pastikan HP & laptop di WiFi yang sama, dan app kamera sedang aktif/running.")
        return

    prev_time = time.time()
    fps_smooth = 0.0

    print("✅ Kamera terhubung. Tekan 'q' pada jendela video untuk keluar.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Gagal membaca frame, mencoba lagi...")
            continue

        results = model.predict(frame, conf=CONF_THRESH, device=DEVICE,
                                 classes=[0], verbose=False)
        r = results[0]
        annotated = r.plot()  # otomatis gambar bounding box + skor

        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now
        fps_smooth = fps_smooth * 0.9 + fps * 0.1  # smoothing biar angka FPS gak lompat-lompat

        n_person = len(r.boxes)
        cv2.putText(annotated, f"FPS: {fps_smooth:.1f} | Orang terdeteksi: {n_person}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Deteksi Manusia Real-Time (YOLOv8n ONNX)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()