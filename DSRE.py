# ===== 完全版（ポップアップ完全排除 + 安定化 + 進捗強化）=====

import os
import sys
import time
import subprocess
import tempfile
import numpy as np
from scipy import signal
import librosa
import resampy
import soundfile as sf

from PySide6 import QtCore, QtWidgets

INPUT_DIR = r"C:\Audio\DSRE"
OUTPUT_DIR = r"C:\Audio\DSRE\Output"

PARAMS = dict(
    m=8,
    decay=1.25,
    pre_hp=3000,
    post_hp=16000,
    target_sr=96000,
    filter_order=11,
    format="FLAC"
)

# ===== ffmpeg（完全非表示）=====
def run_ffmpeg(cmd):
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        startupinfo=si,
        creationflags=subprocess.CREATE_NO_WINDOW
    )

# ===== 保存（本家ロジック維持）=====
def save_wav24_out(in_path, y_out, sr, out_path):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()

    data = y_out.T if y_out.ndim > 1 else y_out[:, None]
    sf.write(tmp.name, data.astype(np.float32), sr, subtype="FLOAT")

    out_path = os.path.splitext(out_path)[0] + ".flac"

    cmd = [
        "ffmpeg","-y",
        "-i", tmp.name,
        "-i", in_path,
        "-map","0:a",
        "-map_metadata","1",
        "-c:a","flac",
        "-sample_fmt","s32",
        out_path
    ]

    run_ffmpeg(cmd)

    os.remove(tmp.name)
    return out_path

# ===== 安定化フィルタ（v2から）=====
def safe_butter(order, cutoff, btype, sr):
    try:
        return signal.butter(order, cutoff/(sr/2), btype)
    except:
        # fallback
        return signal.butter(5, min(cutoff/(sr/2), 0.99), btype)

# ===== DSP（完全維持）=====
def freq_shift_mono(x, f_shift, d_sr):
    N = len(x)
    Np = 1 << int(np.ceil(np.log2(max(1, N))))
    S = signal.hilbert(np.hstack((x, np.zeros(Np - N))))
    factor = np.exp(2j * np.pi * f_shift * d_sr * np.arange(Np))
    return (S * factor)[:N].real

def zansei_impl(x, sr, progress_cb=None, abort_cb=None):
    p = PARAMS

    b,a = safe_butter(p["filter_order"], p["pre_hp"], 'highpass', sr)
    d_src = signal.filtfilt(b,a,x)

    d_res = np.zeros_like(x)
    d_sr = 1.0/sr

    for i in range(p["m"]):
        if abort_cb and abort_cb():
            break

        shift = sr*(i+1)/(p["m"]*2.0)
        d_res += freq_shift_mono(d_src, shift, d_sr) * np.exp(-(i+1)*p["decay"])

        if progress_cb:
            progress_cb(i+1, p["m"])

    b,a = safe_butter(p["filter_order"], p["post_hp"], 'highpass', sr)
    d_res = signal.filtfilt(b,a,d_res)

    adp = float(np.mean(np.abs(d_res)))
    src = float(np.mean(np.abs(x)))
    adj = src/(adp+src+1e-12)

    return (x + d_res) * adj

# ===== Worker =====
class Worker(QtCore.QThread):
    sig_file = QtCore.Signal(int)
    sig_all = QtCore.Signal(int)
    sig_text = QtCore.Signal(str)

    def __init__(self, files):
        super().__init__()
        self.files = files
        self._abort = False
        self._pause = False

    def abort(self):
        self._abort = True

    def pause_toggle(self):
        self._pause = not self._pause

    def run(self):
        total = len(self.files)
        start_time = time.time()

        for i, f in enumerate(self.files, 1):

            if self._abort:
                return

            while self._pause:
                self.msleep(100)

            size = os.path.getsize(f)
            processed = 0

            self.sig_text.emit(f"{i}/{total}")

            y, sr = librosa.load(f, mono=False, sr=None)
            if y.ndim == 1:
                y = y[np.newaxis,:]

            if sr != PARAMS["target_sr"]:
                y = resampy.resample(y, sr, PARAMS["target_sr"])
                sr = PARAMS["target_sr"]

            def step_cb(cur, m):
                pct = int(cur*100/m)
                self.sig_file.emit(pct)

            y_out = zansei_impl(
                y, sr,
                progress_cb=step_cb,
                abort_cb=lambda: self._abort
            )

            base = os.path.basename(f)
            out = os.path.join(OUTPUT_DIR, base)

            save_wav24_out(f, y_out, sr, out)

            # ===== ETA計算 =====
            elapsed = time.time() - start_time
            avg = elapsed / i
            remain = avg * (total - i)

            self.sig_text.emit(f"{i}/{total}  残り{int(remain)}秒")

            self.sig_all.emit(int(i*100/total))
            self.sig_file.emit(100)

# ===== UI =====
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("DSRE")
        self.resize(320,180)

        self.label = QtWidgets.QLabel("待機")
        self.pb_file = QtWidgets.QProgressBar()
        self.pb_all = QtWidgets.QProgressBar()

        self.btn_start = QtWidgets.QPushButton("開始")
        self.btn_pause = QtWidgets.QPushButton("一時停止")
        self.btn_cancel = QtWidgets.QPushButton("取消")

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.pb_file)
        layout.addWidget(self.pb_all)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_pause)
        layout.addWidget(self.btn_cancel)

        self.setLayout(layout)

        self.btn_start.clicked.connect(self.start)
        self.btn_pause.clicked.connect(self.pause)
        self.btn_cancel.clicked.connect(self.cancel)

        self.worker = None

    def load_files(self):
        files = []
        if not os.path.exists(INPUT_DIR):
            return files

        existing = set()
        if os.path.exists(OUTPUT_DIR):
            existing = {os.path.splitext(f)[0] for f in os.listdir(OUTPUT_DIR)}

        for f in os.listdir(INPUT_DIR):
            if f.lower().endswith(".flac"):
                if os.path.splitext(f)[0] not in existing:
                    files.append(os.path.join(INPUT_DIR,f))
        return files

    def start(self):
        files = self.load_files()
        if not files:
            return

        self.worker = Worker(files)
        self.worker.sig_file.connect(self.pb_file.setValue)
        self.worker.sig_all.connect(self.pb_all.setValue)
        self.worker.sig_text.connect(self.label.setText)
        self.worker.start()

    def pause(self):
        if self.worker:
            self.worker.pause_toggle()

    def cancel(self):
        if self.worker:
            self.worker.abort()

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
