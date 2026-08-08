"""Laporan per sesi deteksi manusia + ringkasan Excel.

Meniru mekanisme reports project skripsi (drone_e99_face_recognition):
- CSV per-sesi + manifest JSON sebagai single source,
- Excel summary_detection.xlsx di-build ulang dari source tiap kali (BUKAN append).

Struktur (prefix <ts> = YYYYMMDD_HHMMSS):
  reports/inference/session_<ts>/detection_log_<ts>.csv    (per deteksi)
  reports/inference/session_<ts>/per_second_<ts>.csv       (agregasi per detik)
  reports/inference/session_<ts>/session_stats_<ts>.csv    (ringkasan sesi)
  reports/inference/session_<ts>/persons_per_second_<ts>.png
  reports/runs/<ts>_detection.json                         (manifest)
  reports/summary/summary_detection.xlsx                   (rebuild tiap run)

Pemakaian di main.py:
    rep = SessionReporter(source, model, provider, conf)
    ... tiap frame: rep.add_frame(dets)   # dets (n,5) x1,y1,x2,y2,conf
    rep.finalize(); export_summary()
"""
import csv
import glob
import json
import os
import shutil
import tempfile
import time
from datetime import datetime

import numpy as np
import cv2
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(ROOT, 'reports')
RUNS_DIR = os.path.join(REPORTS_DIR, 'runs')
SUMMARY_DIR = os.path.join(REPORTS_DIR, 'summary')

DETECTION_HEADERS = ['ts', 'frame_idx', 'conf', 'x1', 'y1', 'x2', 'y2', 'fps']
PER_SECOND_HEADERS = ['second', 'frame_count', 'n_detections',
                      'mean_conf', 'max_conf', 'avg_fps']
STATS_FIELDS = ['source', 'model', 'provider', 'conf_thres', 'duration_s',
                'frames', 'total_detections', 'max_persons_in_frame',
                'mean_persons_per_frame', 'mean_conf', 'avg_fps']


def _now_ts():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


class SessionReporter:
    """Pencatat satu sesi live. add_frame() per frame; finalize() menulis laporan."""

    def __init__(self, source, model, provider, conf):
        self.source = source
        self.model = model
        self.provider = provider
        self.conf = conf
        self.ts = _now_ts()
        self.start = time.time()
        self.per_sec = {}   # detik -> [frames, n_deteksi, sum_conf, max_conf, sum_fps]
        self.frames = 0
        self.n_detections = 0
        self.sum_conf = 0.0
        self.max_in_frame = 0
        self.sum_in_frame = 0.0
        self.fps_sum = 0.0
        self.log_path = None
        self._csv = None
        self.session_stats = None

    def add_frame(self, frame_idx, fps, dets):
        """Satu frame diproses. dets: np.ndarray (m,5) x1,y1,x2,y2,conf (boleh kosong)."""
        sec = int(time.time() - self.start)
        rec = self.per_sec.setdefault(sec, [0, 0, 0.0, 0.0, 0.0])
        rec[0] += 1
        rec[4] += fps

        if self._csv is None:
            self._open_log()
        now = datetime.now()
        ts_str = f'{now.strftime("%H:%M:%S")}.{now.microsecond // 1000:03d}'
        for x1, y1, x2, y2, c in dets:
            self._writer.writerow([ts_str, frame_idx, f'{c:.4f}',
                                   f'{x1:.1f}', f'{y1:.1f}',
                                   f'{x2:.1f}', f'{y2:.1f}', f'{fps:.1f}'])
            rec[1] += 1
            rec[2] += float(c)
            rec[3] = max(rec[3], float(c))
            self.n_detections += 1
            self.sum_conf += float(c)

        n = len(dets)
        self.frames += 1
        self.max_in_frame = max(self.max_in_frame, n)
        self.sum_in_frame += n
        self.fps_sum += fps

    def _open_log(self):
        session_dir = os.path.join(REPORTS_DIR, 'inference', f'session_{self.ts}')
        os.makedirs(session_dir, exist_ok=True)
        self.log_path = os.path.join(session_dir, f'detection_log_{self.ts}.csv')
        self._csv = open(self.log_path, 'w', newline='')
        self._writer = csv.writer(self._csv)
        self._writer.writerow(DETECTION_HEADERS)

    def finalize(self):
        """Tutup log + tulis per_second, stats CSV, PNG, manifest. True jika ada data."""
        if self._csv is not None:
            self._csv.close()
            self._csv = None
        if self.frames == 0:
            return False

        session_dir = os.path.join(REPORTS_DIR, 'inference', f'session_{self.ts}')
        per_second_path = os.path.join(session_dir, f'per_second_{self.ts}.csv')
        with open(per_second_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(PER_SECOND_HEADERS)
            for sec in sorted(self.per_sec):
                fc, nd, sc, mx, fs = self.per_sec[sec]
                w.writerow([sec, fc, nd,
                            round(sc / nd, 4) if nd else 0.0,
                            round(mx, 4),
                            round(fs / fc, 2) if fc else 0.0])

        dur = time.time() - self.start
        stats = {
            'source': self.source,
            'model': self.model,
            'provider': self.provider,
            'conf_thres': self.conf,
            'duration_s': round(dur, 2),
            'frames': self.frames,
            'total_detections': self.n_detections,
            'max_persons_in_frame': self.max_in_frame,
            'mean_persons_per_frame': round(self.sum_in_frame / self.frames, 2),
            'mean_conf': round(self.sum_conf / self.n_detections, 4)
                         if self.n_detections else 0.0,
            'avg_fps': round(self.fps_sum / self.frames, 2),
        }
        self.session_stats = stats
        stats_path = os.path.join(session_dir, f'session_stats_{self.ts}.csv')
        with open(stats_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=STATS_FIELDS)
            w.writeheader()
            w.writerow(stats)

        png_path = os.path.join(session_dir, f'persons_per_second_{self.ts}.png')
        _plot_persons(sorted(self.per_sec),
                      [self.per_sec[s][1] for s in sorted(self.per_sec)],
                      png_path, self.source)

        manifest = {
            'timestamp': self.ts,
            'session_id': f'session_{self.ts}',
            'source': self.source,
            'model': self.model,
            'provider': self.provider,
            'conf': self.conf,
            'results': stats,
            # ponytail: basename cukup — folder sesi sudah unik per <ts>
            'files': [os.path.basename(p) for p in
                      (self.log_path, per_second_path, stats_path, png_path)
                      if p is not None],
        }
        os.makedirs(RUNS_DIR, exist_ok=True)
        manifest_path = os.path.join(RUNS_DIR, f'{self.ts}_detection.json')
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        return True


def _plot_persons(secs, series, path, source):
    """Grafik orang/detik pakai cv2 (tanpa matplotlib)."""
    W, H = 1000, 460
    img = np.full((H, W, 3), 255, np.uint8)
    max_y = max(series) or 1
    pad = 70
    if len(secs) < 2:
        secs = [*secs, secs[-1] + 1 if secs else 1]
        series = [*series, 0]
    xs = [pad + (s - min(secs)) * (W - 2 * pad) / (max(secs) - min(secs)) for s in secs]
    ys = [H - 40 - v * (H - 100) / max_y for v in series]
    for i in range(len(secs) - 1):
        x1, y1 = int(xs[i]), int(ys[i])
        x2, y2 = int(xs[i + 1]), int(ys[i + 1])
        cv2.line(img, (x1, y1), (x2, y2), (0, 120, 255), 2)
    step = max(1, len(secs) // 8)
    for i, s in enumerate(secs):
        if i % step == 0:
            cv2.putText(img, str(s), (int(xs[i]) - 10, H - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    cv2.putText(img, f'{source} | orang per detik (max {max_y})',
                (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(img, 'detik', (25, H - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (120, 120, 120), 1)
    cv2.imwrite(path, img)


# ==================== export Excel (rebuild dari source) ================

LOG_HEADERS = ['timestamp', 'session_id', 'source', 'model', 'provider', 'conf',
               'duration_s', 'frames', 'total_detections', 'max_persons_in_frame',
               'mean_persons_per_frame', 'mean_conf', 'avg_fps']
PER_SECOND_HEADERS2 = ['session_id', 'source', 'second', 'frame_count',
                       'n_detections', 'mean_conf', 'max_conf', 'avg_fps']


def _load_manifests():
    out = []
    for path in sorted(glob.glob(os.path.join(RUNS_DIR, '*_detection.json'))):
        try:
            with open(path, encoding='utf-8') as f:
                out.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            print(f'[WARN] Manifest korup, dilewati: {path}')
    return out


def _session_rows():
    rows = []
    for m in _load_manifests():
        r = m.get('results', {})
        rows.append([m.get('timestamp'), m.get('session_id'), m.get('source'),
                     m.get('model'), m.get('provider'), m.get('conf'),
                     r.get('duration_s'), r.get('frames'),
                     r.get('total_detections'), r.get('max_persons_in_frame'),
                     r.get('mean_persons_per_frame'), r.get('mean_conf'),
                     r.get('avg_fps')])
    return rows


def _per_second_rows():
    manifests = {m.get('timestamp'): m for m in _load_manifests()}
    rows = []
    for path in sorted(glob.glob(os.path.join(REPORTS_DIR, 'inference', '**',
                                             'per_second_*.csv'), recursive=True)):
        ts = os.path.basename(path)[len('per_second_'):-len('.csv')]
        m = manifests.get(ts, {})
        sid = m.get('session_id', '')
        src = m.get('source', '')
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                rows.append((sid, src, int(row['second']),
                             int(row['frame_count']), int(row['n_detections']),
                             float(row['mean_conf'] or 0), float(row['max_conf'] or 0),
                             float(row['avg_fps'] or 0)))
    return rows


def _write_table(ws, headers, rows):
    for c, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.font = Font(bold=True)
        cell.fill = PatternFill('solid', fgColor='D9E1F2')
    for r, row in enumerate(rows, 2):
        for c, v in enumerate(row, 1):
            ws.cell(row=r, column=c, value=v)
    for c in range(1, len(headers) + 1):
        vals = [len(str(ws.cell(row=r, column=c).value or ''))
                for r in range(2, len(rows) + 2)]
        ws.column_dimensions[get_column_letter(c)].width = \
            min(max([len(str(headers[c - 1]))] + vals) + 2, 40)
    ws.freeze_panes = 'A2'


def export_summary():
    """Rebuild reports/summary/summary_detection.xlsx dari manifest + CSV."""
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    wb = Workbook()
    default_sheet = wb.active
    if default_sheet is not None:
        wb.remove(default_sheet)

    _write_table(wb.create_sheet('SessionLog'), LOG_HEADERS, _session_rows())

    _write_table(wb.create_sheet('PerSecond'), PER_SECOND_HEADERS2,
                 _per_second_rows())

    ws = wb.create_sheet('PlotsGrid')
    for c, h in enumerate(['Timestamp', 'Source', 'Plot'], 1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.font = Font(bold=True)
        cell.fill = PatternFill('solid', fgColor='D9E1F2')
    ws.column_dimensions['A'].width = 21
    ws.column_dimensions['B'].width = 8
    ws.column_dimensions['C'].width = 80
    manifests = {m.get('timestamp'): m for m in _load_manifests()}
    r = 2
    for path in sorted(glob.glob(os.path.join(REPORTS_DIR, 'inference', '**',
                                             'persons_per_second_*.png'),
                                recursive=True)):
        ts = os.path.basename(path)[len('persons_per_second_'):-len('.png')]
        m = manifests.get(ts, {})
        ws.cell(row=r, column=1, value=ts)
        ws.cell(row=r, column=2, value=m.get('source', ''))
        img = XLImage(path)
        img.width = 480
        img.height = int(img.height * 480 / img.width)
        ws.add_image(img, f'C{r}')
        ws.row_dimensions[r].height = img.height
        r += 1

    out = os.path.join(SUMMARY_DIR, 'summary_detection.xlsx')
    wb.save(out)
    print(f'[DONE] {out}')
    return out


def _check():
    # ponytail: satu run-level check — agregasi + finalize roundtrip
    global REPORTS_DIR, RUNS_DIR, SUMMARY_DIR
    tmp = tempfile.mkdtemp()
    old = (REPORTS_DIR, RUNS_DIR, SUMMARY_DIR)
    REPORTS_DIR, RUNS_DIR, SUMMARY_DIR = (os.path.join(tmp, 'reports'),
                                          os.path.join(tmp, 'reports', 'runs'),
                                          os.path.join(tmp, 'reports', 'summary'))
    try:
        rep = SessionReporter('test', 'best.onnx', 'CUDA', 0.25)
        d1 = np.array([[10, 10, 50, 50, 0.9], [100, 100, 150, 150, 0.8]], np.float32)
        d2 = np.array([[10, 10, 50, 50, 0.7]], np.float32)
        rep.add_frame(0, 30.0, d1)
        rep.add_frame(1, 30.0, d2)
        assert rep.frames == 2
        assert rep.n_detections == 3
        assert rep.max_in_frame == 2
        assert abs(rep.sum_conf - 2.4) < 1e-6  # float32 precision
        rec = list(rep.per_sec.values())[0]
        assert rec[0] == 2 and rec[1] == 3
        assert abs(rec[2] - 2.4) < 1e-6
        assert abs(rec[3] - 0.9) < 1e-6
        assert rep.finalize() is True
        st = rep.session_stats
        assert st is not None
        assert st['mean_persons_per_frame'] == 1.5
        for sub in ('detection_log', 'per_second', 'session_stats'):
            assert glob.glob(os.path.join(REPORTS_DIR, 'inference',
                                          f'session_{rep.ts}',
                                          f'{sub}_{rep.ts}.csv')), sub
        assert os.path.exists(os.path.join(
            REPORTS_DIR, 'inference', f'session_{rep.ts}',
            f'persons_per_second_{rep.ts}.png'))
        assert glob.glob(os.path.join(RUNS_DIR, f'{rep.ts}_detection.json'))
        export_path = export_summary()
        assert os.path.exists(export_path)
        print('[CHECK] SessionReporter agregasi + finalize + export OK')
    finally:
        REPORTS_DIR, RUNS_DIR, SUMMARY_DIR = old
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    _check()
    print('[CHECK] reporting module import OK')