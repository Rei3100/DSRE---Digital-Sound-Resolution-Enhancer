# ===== 完全安定修正版（動作保証＋原因可視化）=====

import os
import sys
import time
import traceback

import subprocess
import soundfile as sf
import tempfile

import numpy as np
from scipy import signal
import librosa
import resampy

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
)

# ===== ffmpeg（完全非表示＋失敗検出）=====
def run_hidden(cmd):
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        startupinfo=si,
        creationflags=subprocess.CREATE_NO_WINDOW
    )

    if result.returncode != 0:
        raise RuntimeError("ffmpeg失敗:\n" + result.stderr.decode(errors="ignore"))

# ===== 安全読み込み =====
def load_audio_safe(path):
    try:
        data, sr = sf.read(path, always_2d=True)
        return data.T, sr
    except:
        y, sr = librosa.load(path, mono=False, sr=None)
        if y.ndim == 1:
            y = y[np.newaxis, :]
        return y, sr

# ===== 保存 =====
def save_wav24_out(in_path, y_out, sr, out_path):

    if y_out.ndim == 1:
        data = y_out[:, None]
    else:
        data = y_out.T if y_out.shape[0] < y_out.shape[1] else y_out

    data = data.astype(np.float32, copy=False)

    peak = float(np.max(np.abs(data)))
    if peak > 1.0:
        data /= peak

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    sf.write(tmp.name, data, sr, subtype="FLOAT")

    out_path = os.path.splitext(out_path)[0] + ".flac"

    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-i", tmp.name,
        "-i", in_path,
        "-map", "0:a",
        "-map_metadata", "1",
        "-c:a", "flac",
        "-sample_fmt", "s32",
        out_path
    ]

    run_hidden(cmd)
    os.remove(tmp.name)

# ===== DSP（変更なし）=====
def freq_shift_mono(x, f_shift, d_sr):
    N = len(x)
    Np = 1 << int(np.ceil(np.log2(max(1, N))))
    S = signal.hilbert(np.hstack((x, np.zeros(Np - N))))
    F = np.exp(2j * np.pi * f_shift * d_sr * np.arange(Np))
    return (S * F)[:N].real

def freq_shift_multi(x, f_shift, d_sr):
    return np.asarray([freq_shift_mono(ch, f_shift, d_sr) for ch in x])

def safe_butter(order, cutoff, sr):
    nyq = sr / 2
    cutoff = min(cutoff, nyq * 0.95)
    order = min(order, 20)
    return signal.butter(order, cutoff / nyq, 'highpass')

def zansei_impl(x, sr, progress_cb=None, abort_cb=None):

    b, a = safe_butter(PARAMS["filter_order"], PARAMS["pre_hp"], sr)
    d_src = signal.filtfilt(b, a, x)

    d_sr = 1.0 / sr
    f_dn = freq_shift_mono if (x.ndim == 1) else freq_shift_multi
    d_res = np.zeros_like(x)

    total = PARAMS["m"]

    for i in range(total):

        if abort_cb and abort_cb():
            return x  # ← 安全終了

        shift = sr * (i + 1) / (total * 2.0)
        d_res += f_dn(d_src, shift, d_sr) * np.exp(-(i + 1) * PARAMS["decay"])

        if progress_cb:
            progress_cb(i + 1, total)

    b, a = safe_butter(PARAMS["filter_order"], PARAMS["post_hp"], sr)
    d_res = signal.filtfilt(b, a, d_res)

    adp = float(np.mean(np.abs(d_res)))
    src = float(np.mean(np.abs(x)))

    adj = src / (adp + src + 1e-12)

    return (x + d_res) * adj

# ===== Worker =====
class Worker(QtCore.QThread):
    sig_step = QtCore.Signal(int)
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
        start = time.time()

        for idx, path in enumerate(self.files, 1):

            if self._abort:
                return

            while self._pause:
                self.msleep(100)

            try:
                y, sr = load_audio_safe(path)

                if sr != PARAMS["target_sr"]:
                    y = resampy.resample(y, sr, PARAMS["target_sr"])
                    sr = PARAMS["target_sr"]

                def step_cb(cur, m):
                    self.sig_step.emit(int(cur * 100 / m))

                y_out = zansei_impl(
                    y, sr,
                    progress_cb=step_cb,
                    abort_cb=lambda: self._abort
                )

                out = os.path.join(OUTPUT_DIR, os.path.basename(path))
                save_wav24_out(path, y_out, sr, out)

            except Exception as e:
                self.sig_text.emit("エラー:\n" + str(e))
                continue

            elapsed = time.time() - start
            remain = (elapsed / idx) * (total - idx)

            self.sig_text.emit(f"{idx}/{total} 残り{int(remain)}秒")

            self.sig_all.emit(int(idx * 100 / total))
            self.sig_step.emit(100)

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

        existing = set()
        if os.path.exists(OUTPUT_DIR):
            existing = {os.path.splitext(f)[0] for f in os.listdir(OUTPUT_DIR)}

        for f in os.listdir(INPUT_DIR):
            if f.lower().endswith(".flac"):
                if os.path.splitext(f)[0] not in existing:
                    files.append(os.path.join(INPUT_DIR, f))

        return files

    def start(self):
        files = self.load_files()
        if not files:
            return

        self.worker = Worker(files)
        self.worker.sig_step.connect(self.pb_file.setValue)
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
