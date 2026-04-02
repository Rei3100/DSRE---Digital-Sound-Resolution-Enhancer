# ===== 完全修正版（処理完全維持＋ポップアップ除去＋進捗復元）=====

import os
import sys
import traceback
import time
from typing import Optional

import subprocess
import soundfile as sf
import tempfile

import numpy as np
from scipy import signal
import librosa
import resampy

from PySide6 import QtCore, QtWidgets

# ===== 固定 =====
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

# ===== ffmpeg非表示 =====
def run_hidden(cmd):
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    return subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        startupinfo=si,
        creationflags=subprocess.CREATE_NO_WINDOW
    )

# ===== 元そのまま保存処理 =====
def save_wav24_out(in_path, y_out, sr, out_path, fmt="FLAC", normalize=True):

    if y_out.ndim == 1:
        data = y_out[:, None]
    else:
        data = y_out.T if y_out.shape[0] < y_out.shape[1] else y_out

    data = data.astype(np.float32, copy=False)

    if normalize:
        peak = float(np.max(np.abs(data)))
        if peak > 1.0:
            data /= peak
    else:
        data = np.clip(data, -1.0, 1.0)

    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_wav.close()
    sf.write(tmp_wav.name, data, sr, subtype="FLOAT")

    out_path = os.path.splitext(out_path)[0] + ".flac"

    cmd = [
        "ffmpeg", "-y",
        "-i", tmp_wav.name,
        "-i", in_path,
        "-map", "0:a",
        "-map_metadata", "1",
        "-c:a", "flac",
        "-sample_fmt", "s32",
        out_path
    ]

    run_hidden(cmd)

    os.remove(tmp_wav.name)
    return out_path

# ===== 元DSP完全維持 =====
def freq_shift_mono(x, f_shift, d_sr):
    N_orig = len(x)
    N_padded = 1 << int(np.ceil(np.log2(max(1, N_orig))))
    S_hilbert = signal.hilbert(np.hstack((x, np.zeros(N_padded - N_orig, dtype=x.dtype))))
    S_factor = np.exp(2j * np.pi * f_shift * d_sr * np.arange(0, N_padded))
    return (S_hilbert * S_factor)[:N_orig].real

def freq_shift_multi(x, f_shift, d_sr):
    return np.asarray([freq_shift_mono(x[i], f_shift, d_sr) for i in range(len(x))])

def zansei_impl(
    x, sr,
    m, decay, pre_hp, post_hp, filter_order,
    progress_cb=None,
    abort_cb=None,
):
    b, a = signal.butter(filter_order, pre_hp / (sr / 2), 'highpass')
    d_src = signal.filtfilt(b, a, x)

    d_sr = 1.0 / sr
    f_dn = freq_shift_mono if (x.ndim == 1) else freq_shift_multi
    d_res = np.zeros_like(x)

    for i in range(m):
        if abort_cb and abort_cb():
            break

        shift_hz = sr * (i + 1) / (m * 2.0)
        d_res += f_dn(d_src, shift_hz, d_sr) * np.exp(-(i + 1) * decay)

        if progress_cb:
            progress_cb(i + 1, m)

    b, a = signal.butter(filter_order, post_hp / (sr / 2), 'highpass')
    d_res = signal.filtfilt(b, a, d_res)

    adp_power = float(np.mean(np.abs(d_res)))
    src_power = float(np.mean(np.abs(x)))
    adj_factor = src_power / (adp_power + src_power + 1e-12)

    return (x + d_res) * adj_factor

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
        start_time = time.time()

        for idx, path in enumerate(self.files, 1):

            if self._abort:
                return

            while self._pause:
                self.msleep(100)

            self.sig_text.emit(f"{idx}/{total}")

            y, sr = librosa.load(path, mono=False, sr=None)

            if y.ndim == 1:
                y = y[np.newaxis, :]

            if sr != PARAMS["target_sr"]:
                y = resampy.resample(y, sr, PARAMS["target_sr"])
                sr = PARAMS["target_sr"]

            def step_cb(cur, m):
                pct = int(cur * 100 / m)
                self.sig_step.emit(pct)

            y_out = zansei_impl(
                y, sr,
                PARAMS["m"],
                PARAMS["decay"],
                PARAMS["pre_hp"],
                PARAMS["post_hp"],
                PARAMS["filter_order"],
                progress_cb=step_cb,
                abort_cb=lambda: self._abort
            )

            base = os.path.basename(path)
            out = os.path.join(OUTPUT_DIR, base)

            save_wav24_out(path, y_out, sr, out)

            elapsed = time.time() - start_time
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
        self.pb_file.setRange(0,100)

        self.pb_all = QtWidgets.QProgressBar()
        self.pb_all.setRange(0,100)

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
