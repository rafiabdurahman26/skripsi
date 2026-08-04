import time
import av
import cv2
from threading import Thread, Lock
from djitellopy import Tello


class VideoDecoder:
    def __init__(self, port=11111):
        self._frame = None
        self._lock = Lock()
        self._running = False
        self._thread = None
        self._container = None
        self._address = f'udp://@0.0.0.0:{port}'

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = Thread(target=self._decode, daemon=True)
        self._thread.start()

    def _decode(self):
        try:
            self._container = av.open(self._address, timeout=(5, None))
            for frame in self._container.decode(video=0):
                if not self._running:
                    break
                with self._lock:
                    self._frame = frame.to_ndarray(format='bgr24')
        except Exception:
            pass
        finally:
            if self._container:
                try:
                    self._container.close()
                except Exception:
                    pass

    @property
    def frame(self):
        with self._lock:
            return self._frame

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)


class Drone:
    def __init__(self):
        self.tello = Tello()
        self._is_flying = False
        self._battery = 0
        self._bt = 0
        self._decoder = None

    @property
    def is_flying(self):
        return self._is_flying

    def connect(self):
        self.tello.connect()
        self.tello.streamon()
        self._decoder = VideoDecoder(port=self.tello.vs_udp_port)
        self._decoder.start()

    def takeoff(self):
        self.tello.takeoff()
        self._is_flying = True

    def land(self):
        self.tello.land()
        self._is_flying = False

    def toggle_flight(self):
        self.land() if self._is_flying else self.takeoff()

    def send_rc(self, lr, fb, ud, yaw):
        self.tello.send_rc_control(lr, fb, ud, yaw)

    def get_frame(self):
        return self._decoder.frame if self._decoder else None

    def get_height(self):
        return self.tello.get_height()

    def get_flight_time(self):
        return self.tello.get_flight_time()

    def get_battery(self):
        now = time.time()
        if now - self._bt > 2:
            try:
                self._battery = self.tello.get_battery()
            except RuntimeError:
                pass
            self._bt = now
        return self._battery

    def disconnect(self):
        if self._decoder:
            self._decoder.stop()
            self._decoder = None
        if self._is_flying:
            self.land()
        self.tello.streamoff()
        self.tello.end()
