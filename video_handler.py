import cv2
import math
import os
import time
from datetime import datetime
from config import PHOTO_DIR, VIDEO_DIR, BATTERY_WARN, BATTERY_CRITICAL


class VideoHandler:
    def __init__(self):
        os.makedirs(PHOTO_DIR, exist_ok=True)
        os.makedirs(VIDEO_DIR, exist_ok=True)
        self._recording = False
        self._rec_start = 0
        self._writer = None
        self.photo_count = 0

    @property
    def recording(self):
        return self._recording

    def draw_drone_state(self, frame, lr, fb, ud, yaw):
        h, w = frame.shape[:2]
        cx, cy = 100, h - 100

        bg = frame[cy-40:cy+40, cx-40:cx+40].copy()
        cv2.rectangle(bg, (0, 0), (80, 80), (0, 0, 0), -1)
        cv2.addWeighted(bg, 0.35, frame[cy-40:cy+40, cx-40:cx+40], 0.65, 0, frame[cy-40:cy+40, cx-40:cx+40])

        cv2.circle(frame, (cx, cy), 8, (180, 180, 180), 2)
        cv2.line(frame, (cx-6, cy-6), (cx+6, cy+6), (140, 140, 140), 1)
        cv2.line(frame, (cx+6, cy-6), (cx-6, cy+6), (140, 140, 140), 1)

        def _draw(val, ox, oy, dx, dy):
            col = (0, 0, 255) if abs(val) > 0.05 else (50, 50, 50)
            mag = int(min(abs(val), 1.0) * 25)
            if mag < 2:
                return
            ex = ox + dx * mag
            ey = oy + dy * mag
            cv2.arrowedLine(frame, (ox, oy), (ex, ey), col, 2, tipLength=0.3)

        _draw(fb, cx, cy-12, 0, -1)     # forward
        _draw(fb, cx, cy+12, 0, 1)      # backward
        _draw(lr, cx-12, cy, -1, 0)     # left
        _draw(lr, cx+12, cy, 1, 0)      # right
        _draw(ud, cx+25, cy-8, 0, -1)   # up
        _draw(ud, cx+25, cy+8, 0, 1)    # down
        _draw(yaw, cx-25, cy-8, -1, 0)  # yaw left
        _draw(yaw, cx-25, cy+8, 1, 0)   # yaw right

    def render(self, frame, battery, flying, mode, rec, trim_lr, speed_pct=100, grid=False, height=0, flight_time=0, lr=0.0, fb=0.0, ud=0.0, yaw=0.0):
        h, w = frame.shape[:2]
        bar = frame[:52].copy()
        cv2.rectangle(bar, (0, 0), (w, 52), (0, 0, 0), -1)
        cv2.addWeighted(bar, 0.55, frame[:52], 0.45, 0, frame[:52])

        cv2.putText(frame, mode.upper(), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        bat_col = (0, 0, 255) if battery <= BATTERY_CRITICAL else (
            (0, 255, 255) if battery <= BATTERY_WARN else (255, 255, 255)
        )
        cv2.putText(frame, f'BAT {battery}%', (10, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bat_col, 1)

        status = 'FLY' if flying else 'GRD'
        col = (0, 255, 0) if flying else (0, 0, 255)
        cv2.putText(frame, status, (160, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        cv2.putText(frame, f'PH {self.photo_count}', (160, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, f'TR {trim_lr:+d}', (320, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, f'SPD {speed_pct}%', (320, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f'ALT {height / 100:.1f}m', (440, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f'TM {flight_time // 60:02d}:{flight_time % 60:02d}', (560, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if rec:
            elapsed = int(time.time() - self._rec_start)
            cv2.circle(frame, (w - 40, 20), 6, (0, 0, 255), -1)
            cv2.putText(frame, 'REC', (w - 75, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            cv2.putText(frame, f'{elapsed // 60:02d}:{elapsed % 60:02d}',
                        (w - 78, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        if grid:
            for i in (1, 2):
                cv2.line(frame, (w * i // 3, 0), (w * i // 3, h), (180, 180, 180), 1)
                cv2.line(frame, (0, h * i // 3), (w, h * i // 3), (180, 180, 180), 1)

        self.draw_drone_state(frame, lr, fb, ud, yaw)

        if battery <= BATTERY_CRITICAL:
            red = frame.copy()
            cv2.rectangle(red, (0, 0), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(red, 0.2, frame, 0.8, 0, frame)
            text = 'BATTERY CRITICAL'
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            x = (w - size[0]) // 2
            y = h // 2
            cv2.putText(frame, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif battery <= BATTERY_WARN:
            cv2.putText(frame, 'LOW BATTERY', (w - 130, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return frame

    def capture_photo(self, frame):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        cv2.imwrite(f'{PHOTO_DIR}/tello_{ts}.jpg', frame)
        self.photo_count += 1

    def toggle_recording(self, frame_shape):
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording(frame_shape)

    def _start_recording(self, shape):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h, w = shape[:2]
        self._writer = cv2.VideoWriter(
            f'{VIDEO_DIR}/tello_{ts}.mp4', fourcc, 20.0, (w, h)
        )
        self._recording = True
        self._rec_start = time.time()

    def _stop_recording(self):
        if self._writer:
            self._writer.release()
            self._writer = None
        self._recording = False

    def write_frame(self, frame):
        if self._writer:
            self._writer.write(frame)
