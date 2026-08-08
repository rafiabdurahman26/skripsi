"""
Deteksi manusia YOLOv8 (ONNX) murni dengan onnxruntime — tanpa torch/ultralytics.
Auto provider: CUDA (NVIDIA) -> CPU fallback. Self-check: python detector.py
"""

import os
import numpy as np
import cv2

os.environ.setdefault('ORT_LOG_LEVEL', '3')  # redam log warning native CUDA di konsol
import onnxruntime as ort
# ponytail: hanya supress sebatas proses ini; glog default ORT = 3 (error-only)
try:
    ort.set_default_logger_severity(3)
except Exception:
    pass


class Detector:
    def __init__(self, model_path, device='auto', conf=0.35, imgsz=640, raw_input=True):
        self.model_path = str(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.raw_input = raw_input  # True: feed 0..255 mentah; False: bagi 255
        self.session = self._create_session(device)
        self.in_name = self.session.get_inputs()[0].name
        self.out_name = self.session.get_outputs()[0].name
        # ponytail: buffer reuse - blob 640x640x3 float32 (4.9MB) dialokasi sekali
        self.canvas = None
        self.blob = np.empty((1, 3, imgsz, imgsz), np.float32)

    @property
    def device(self):
        return self.session.get_providers()[0]

    @property
    def label(self):
        # ponytail: 2 nilai saja (CUDA/CPU), tidak perlu enum
        return 'CUDA' if 'CUDA' in self.session.get_providers()[0] else 'CPU'

    def _create_session(self, device):
        available = ort.get_available_providers()
        want_cuda = device == 'cuda' or (device == 'auto' and 'CUDAExecutionProvider' in available)
        if want_cuda:
            try:
                # Memuat DLL CUDA/cuDNN dari NVIDIA site-packages di dalam venv.
                # Tanpa menyentuh toolkit CUDA 12.1 sistem / PATH / project lain.
                ort.preload_dlls()
                so = ort.SessionOptions()
                so.log_severity_level = 3  # cuma error, redam warning Conv di konsol
                cuda_opts = {
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,   # batas aman untuk VRAM 4GB
                    'arena_extend_strategy': 'kSameAsRequested',
                    # EXHAUSTIVE: cuDNN pilih algo GPU terbaik per layer.
                    # DEFAULT memakai jalur lambat (30ms -> 9.5ms, 3.2x).
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                }
                return ort.InferenceSession(
                    self.model_path, so,
                    providers=[('CUDAExecutionProvider', cuda_opts), 'CPUExecutionProvider'],
                )
            except Exception as e:
                print(f'[DETECT] [!] CUDA gagal ({e}) - fallback CPU')
        return ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])

    def _preprocess(self, frame):
        h, w = frame.shape[:2]
        scale = self.imgsz / max(h, w)
        nw, nh = round(w * scale), round(h * scale)
        img = cv2.resize(frame, (nw, nh))
        # ponytail: cache canvas letterbox; alokasi ulang tiap frame hanya ~1ms CPU
        if self.canvas is None:
            self.canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.float32)
        canvas = self.canvas
        canvas[:nh, :nw] = img
        self.blob[0] = canvas[:, :, ::-1].transpose(2, 0, 1)
        if not self.raw_input:
            self.blob /= 255.0  # in-place, tanpa alokasi baru
        return self.blob, 1.0 / scale

    def detect(self, frame):
        """Return np.ndarray (n,5): x1, y1, x2, y2, conf."""
        blob, inv_scale = self._preprocess(frame)
        out = self.session.run([self.out_name], {self.in_name: blob})[0]
        if len(out.shape) == 3 and out.shape[2] == 6:
            return self._postprocess_boxes(out, inv_scale)   # NMS sudah di graph
        return self._postprocess_grid(out, inv_scale)        # (1, 5+cls, N)

    def _postprocess_grid(self, out, inv_scale):
        # out: (1, C, N) = [x, y, w, h, conf...] per anchor (class rows mengikuti)
        pred = out[0]
        confs = pred[4:].max(axis=0)          # multi-class aman, single-class pas
        conf_mask = confs >= self.conf
        n = int(conf_mask.sum())
        if n == 0:
            return np.empty((0, 5), np.float32)
        xywh = pred[:4, conf_mask].T * inv_scale
        confs = confs[conf_mask]
        keep = self._nms(xywh, confs)
        return self._xy(xywh, confs)[keep]

    def _postprocess_boxes(self, out, inv_scale):
        # out: (1, M, 6) = x1,y1,x2,y2,conf,cls (NMS sudah di dalam graph)
        boxes = out[0]
        boxes = boxes[boxes[:, 4] >= self.conf]
        if len(boxes) == 0:
            return np.empty((0, 5), np.float32)
        keep = self._nms_xyxy(boxes[:, :4], boxes[:, 4])
        boxes = boxes[keep]
        return np.column_stack([boxes[:, 0], boxes[:, 1], boxes[:, 2],
                                boxes[:, 3], boxes[:, 4]]) * \
            np.array([inv_scale, inv_scale, inv_scale, inv_scale, 1.0])

    @staticmethod
    def _nms_xyxy(xyxy, confs, iou_thres=0.45):
        xywh = np.column_stack([(xyxy[:, 0] + xyxy[:, 2]) / 2,
                                (xyxy[:, 1] + xyxy[:, 3]) / 2,
                                xyxy[:, 2] - xyxy[:, 0],
                                xyxy[:, 3] - xyxy[:, 1]])
        return Detector._nms(xywh, confs, iou_thres)

    @staticmethod
    def _nms(xywh, confs, iou_thres=0.45):
        x1 = xywh[:, 0] - xywh[:, 2] / 2
        y1 = xywh[:, 1] - xywh[:, 3] / 2
        x2 = xywh[:, 0] + xywh[:, 2] / 2
        y2 = xywh[:, 1] + xywh[:, 3] / 2
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(confs)[::-1]
        keep = []
        while order.size:
            i = order[0]
            keep.append(i)
            rest = order[1:]
            if rest.size:
                xx1 = np.maximum(x1[i], x1[rest])
                yy1 = np.maximum(y1[i], y1[rest])
                xx2 = np.minimum(x2[i], x2[rest])
                yy2 = np.minimum(y2[i], y2[rest])
                iw = np.maximum(0.0, xx2 - xx1 + 1)
                ih = np.maximum(0.0, yy2 - yy1 + 1)
                inter = iw * ih
                iou = inter / (area[i] + area[rest] - inter + 1e-6)
                order = rest[iou < iou_thres]
            else:
                order = rest
        return np.array(keep, dtype=np.int64)

    @staticmethod
    def _xy(xywh, confs):
        x1 = xywh[:, 0] - xywh[:, 2] / 2
        y1 = xywh[:, 1] - xywh[:, 3] / 2
        x2 = xywh[:, 0] + xywh[:, 2] / 2
        y2 = xywh[:, 1] + xywh[:, 3] / 2
        return np.column_stack([x1, y1, x2, y2, confs])

    def annotate(self, frame, dets=None, color=(0, 255, 0)):
        """Gambar bounding box. Return (frame_annotated, count)."""
        if dets is None:  # ponytail: dets opsional agar main.py tak double-detect
            dets = self.detect(frame)
        for x1, y1, x2, y2, c in dets:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f'{c:.2f}', (int(x1), max(0, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame, len(dets)


def _check():
    # ponytail: satu runnable check - NMS harus suppress box overlap (satu objek)
    xywh = np.array([[100, 100, 50, 50], [105, 105, 50, 50], [500, 500, 40, 40]], np.float32)
    confs = np.array([0.9, 0.8, 0.7], np.float32)
    keep = Detector._nms(xywh, confs)
    assert len(keep) == 2, f'NMS overlap tidak ter-suppress: {keep}'
    dets = Detector._xy(xywh, confs)[keep]
    assert np.allclose(dets[:, 4], [0.9, 0.7], rtol=1e-5), 'konfidensi hasil tidak urut'
    print('[CHECK] NMS OK')

    # layout export-NMS (1, M, 6): baris = x1,y1,x2,y2,conf,cls
    d = object.__new__(Detector)
    d.conf = 0.25
    out = np.zeros((1, 300, 6), np.float32)
    out[0, 0] = [10, 10, 60, 60, 0.9, 0]
    out[0, 1] = [12, 12, 58, 58, 0.8, 0]   # overlap, harus di-NMS
    out[0, 2] = [400, 400, 460, 460, 0.3, 0]
    dets = d._postprocess_boxes(out, 2.0)  # inv_scale 2 -> koordinat digandakan
    assert len(dets) == 2, f'boxes layout gagal: {dets}'
    assert np.allclose(dets[0][:2], [20, 20]), 'skala inv_scale tidak diterapkan'
    assert np.allclose(dets[0][4], 0.9), 'conf salah'
    print('[CHECK] Layout (1,M,6) OK')


if __name__ == '__main__':
    _check()
    print('[CHECK] Detector module import OK')
