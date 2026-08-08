"""
Deteksi manusia real-time terintegrasi DJI Tello.
Mode:
  --source tello  (default)  kontrol drone + overlay deteksi manusia
  --source phone             fallback kamera HP (IP Webcam) - deteksi saja

Jalankan:
  uv run python main.py --source tello
  uv run python main.py --source phone
Tekan 'q'/'ESC' untuk keluar.
"""

import argparse
import math
import os
import sys
import time

import cv2

from detector import Detector


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description='Deteksi manusia + DJI Tello')
    p.add_argument('--source', choices=['tello', 'phone'], default='tello')
    p.add_argument('--model', default=os.path.join(here, 'best_model.onnx'))
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--input', choices=['raw', 'norm'], default='raw',
                   help='raw: feed 0-255 (best_model.onnx); norm: bagi 255 (model gaya ultralytics)')
    p.add_argument('--phone-url', default='http://192.168.251.201:8080/video')
    return p.parse_args()


def fps_meter():
    prev = time.time()
    smooth = 0.0
    while True:
        now = time.time()
        fps = 1.0 / (now - prev) if now != prev else 0.0
        prev = now
        smooth = smooth * 0.9 + fps * 0.1
        yield smooth


def run_phone(args, detector):
    from reporting import SessionReporter, export_summary
    cap = cv2.VideoCapture(args.phone_url)
    if not cap.isOpened():
        print(f'[PHONE] [!] Tidak bisa membuka stream: {args.phone_url}')
        print('        Pastikan HP & laptop di WiFi yang sama dan app IP Webcam aktif.')
        return 1
    print(f'[PHONE] Stream OK. Tekan q untuk keluar.')
    rep = SessionReporter(args.source, os.path.basename(args.model),
                          detector.label, args.conf)
    fps = fps_meter()
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame_idx += 1
            fpsv = next(fps)
            dets = detector.detect(frame)
            frame, n = detector.annotate(frame, dets)
            rep.add_frame(frame_idx, fpsv, dets)
            cv2.putText(frame, f'FPS: {fpsv:.1f} | Orang: {n} | {detector.label}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('Deteksi Manusia (YOLOv8n ONNX)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        _finish_report(rep)
    return 0


def _finish_report(rep):
    """Tutup sesi laporan + rebuild Excel. Gagal laporan tidak menggagalkan run."""
    try:
        if rep.finalize():
            print(f'[REPORT] Laporan sesi tersimpan ({rep.ts})')
        from reporting import export_summary
        export_summary()
    except Exception as e:
        print(f'[REPORT] [!] Gagal menyimpan laporan: {e}')


def run_tello(args, detector):
    from config import TRIM_STEP, TRIM_MAX, SPEED_MODES, BATTERY_CRITICAL, load_config, save_config
    from drone import Drone
    from input_handler import InputHandler
    from video_handler import VideoHandler
    from reporting import SessionReporter

    print_help()
    drone = Drone()
    inp = InputHandler()
    video = VideoHandler()
    rep = SessionReporter(args.source, os.path.basename(args.model),
                          detector.label, args.conf)
    frame_idx = 0

    if inp.has_gamepad():
        print('[OK] Gamepad terdeteksi')
    else:
        print('[!] Tidak ada gamepad - keyboard only')

    try:
        drone.connect()
        print(f'[OK] Terhubung - Baterai: {drone.get_battery()}%')
    except Exception as e:
        print(f'[FAIL] Gagal konek: {e}')
        return 1

    trim_lr, speed_idx = load_config()
    print(f'[CONFIG] trim={trim_lr:+d}, speed={SPEED_MODES[speed_idx]}%')
    frame = None
    show_grid = False
    last_rc_print = 0.0
    fps = fps_meter()

    try:
        while True:
            key = cv2.waitKey(1) & 0xFF
            st = inp.poll(key)

            if st.quit:
                break
            if st.switch_mode:
                inp.switch_mode()
                print(f'[MODE] Beralih ke {inp.mode.upper()}')
            if st.emergency_land and drone.is_flying:
                try:
                    drone.land()
                    print('[EMERGENCY] Landed!')
                except Exception as e:
                    print(f'[!] Emergency land gagal: {e}')
            elif st.takeoff_land:
                try:
                    drone.toggle_flight()
                    inp.vibrate(32768, 32768, 0.2)
                except Exception as e:
                    print(f'[!] Takeoff/Land gagal: {e}')
            if st.speed_up:
                speed_idx = (speed_idx + 1) % len(SPEED_MODES)
                print(f'[SPEED] {SPEED_MODES[speed_idx]}%')
            if st.speed_down:
                speed_idx = (speed_idx - 1) % len(SPEED_MODES)
                print(f'[SPEED] {SPEED_MODES[speed_idx]}%')

            if st.photo:
                if frame is not None:
                    video.capture_photo(frame)
                    inp.vibrate(65535, 0, 0.1)
                    print(f'[PHOTO] Tersimpan ({video.photo_count} total)')
            if st.record_toggle:
                if frame is not None and not video.recording:
                    video.toggle_recording(frame.shape)
                elif video.recording:
                    video.toggle_recording(None)
                inp.vibrate(0, 65535, 0.3)
                print(f'[REC] {"Start" if video.recording else "Stop"}')
            if st.trim_left:
                trim_lr = clamp(trim_lr - TRIM_STEP, -TRIM_MAX, TRIM_MAX)
                print(f'[TRIM] LR={trim_lr:+d}')
            if st.trim_right:
                trim_lr = clamp(trim_lr + TRIM_STEP, -TRIM_MAX, TRIM_MAX)
                print(f'[TRIM] LR={trim_lr:+d}')
            if st.trim_reset:
                trim_lr = 0
                print('[TRIM] Reset')
            if key == ord('g'):
                show_grid = not show_grid
                print(f'[GRID] {"On" if show_grid else "Off"}')

            spd = SPEED_MODES[speed_idx] / 100.0
            lr = clamp(int(rate_curve(st.lr) * 100 * spd) + trim_lr, -100, 100)
            fb = clamp(int(-rate_curve(st.fb) * 100 * spd), -100, 100)
            ud = clamp(int(-rate_curve(st.ud) * 100 * spd), -100, 100)
            yaw = clamp(int(-rate_curve(st.yaw) * 100 * spd), -100, 100)

            if any((lr, fb, ud, yaw)) and time.time() - last_rc_print > 0.5:
                print(f'[RC] lr={lr:4d} fb={fb:4d} ud={ud:4d} yaw={yaw:4d}')
                last_rc_print = time.time()

            drone.send_rc(lr, fb, ud, yaw)

            battery = drone.get_battery()
            if battery <= BATTERY_CRITICAL and drone.is_flying:
                try:
                    drone.land()
                    print('[AUTO-LAND] Baterai kritis - mendarat')
                except Exception as e:
                    print(f'[!] Auto-land gagal: {e}')

            frame = drone.get_frame()
            if frame is not None:
                frame_idx += 1
                fpsv = next(fps)
                dets = detector.detect(frame)
                frame, n = detector.annotate(frame, dets)
                rep.add_frame(frame_idx, fpsv, dets)
                overlay = video.render(
                    frame, battery, drone.is_flying, inp.mode, video.recording,
                    trim_lr, SPEED_MODES[speed_idx], show_grid,
                    drone.get_height(), drone.get_flight_time(), lr, fb, ud, yaw,
                )
                h, w = overlay.shape[:2]
                cv2.putText(overlay, f'PERSON {n} | FPS {fpsv:.0f} | {detector.label}',
                            (w - 230, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)
                video.write_frame(overlay)
                cv2.imshow('Tello + Deteksi Manusia', overlay)

            time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        save_config(trim_lr, speed_idx)
        drone.disconnect()
        cv2.destroyAllWindows()
        _finish_report(rep)
        print('[DONE] Disconnected')
    return 0


def print_help():
    print('=' * 50)
    print('DJI Tello + Deteksi Manusia (YOLOv8n ONNX)')
    print('=' * 50)
    print('  W/A/S/D      Gerak maju/kiri/mundur/kanan')
    print('  Panah ^/v    Naik/Turun')
    print('  Panah </>    Yaw kiri/kanan')
    print('  SPACE        Takeoff / Land (toggle)')
    print('  Q            Ambil foto')
    print('  E            Mulai/hentikan rekaman')
    print('  [ / ]        Trim kiri/kanan')
    print('  TAB          Reset trim')
    print('  R            Ganti mode input')
    print('  C / X        Speed naik/turun')
    print('  F            Emergency land')
    print('  G            Grid rule of thirds')
    print('  ESC          Keluar')
    print('=' * 50)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def rate_curve(x):
    return math.copysign(abs(x) ** 3, x)


def main():
    args = parse_args()
    if not os.path.exists(args.model):
        print(f'[DETECT] [!] Model tidak ditemukan: {args.model}')
        return 1
    detector = Detector(args.model, device=args.device, conf=args.conf,
                        raw_input=args.input == 'raw')
    print(f'[DETECT] Model dimuat | provider: {detector.device}')

    if args.source == 'tello':
        return run_tello(args, detector)
    return run_phone(args, detector)


if __name__ == '__main__':
    sys.exit(main())