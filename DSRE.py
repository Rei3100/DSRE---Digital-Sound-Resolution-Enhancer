# ==== 全置き換え（元処理維持版）====

import os
import sys
import subprocess
import tempfile
import numpy as np
from scipy import signal
import librosa
import resampy
import soundfile as sf

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
)

# ===== ffmpeg =====
def add_ffmpeg_to_path():
    if hasattr(sys, "_MEIPASS"):
        ffmpeg_dir = os.path.join(sys._MEIPASS, "ffmpeg")
    else:
        ffmpeg_dir = os.path.join(os.path.dirname(__file__), "ffmpeg")
    os.environ["PATH"] += os.pathsep + ffmpeg_dir

add_ffmpeg_to_path()

# ===== 保存（元準拠）=====
def save_out(in_path, y_out, sr, out_path):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()

    sf.write(tmp.name, y_out.T if y_out.ndim > 1 else y_out, sr, subtype="FLOAT")

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

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW
    )

    os.remove(tmp.name)
    return out_path

# ===== 元DSP復元 =====
def freq_shift_mono(x, f_shift, d_sr):
    N = len(x)
    Np = 1 << int(np.ceil(np.log2(max(1, N))))
    S = signal.hilbert(np.hstack((x, np.zeros(Np - N))))
    factor = np.exp(2j * np.pi * f_shift * d_sr * np.arange(Np))
    return (S * factor)[:N].real

def zansei_impl(x, sr, progress_cb=None):
    p = PARAMS

    b,a = signal.butter(p["filter_order"], p["pre_hp"]/(sr/2),'highpass')
    d_src = signal.filtfilt(b,a,x)

    d_res = np.zeros_like(x)

    for i in range(p["m"]):
        shift = sr*(i+1)/(p["m"]*2.0)
        d_res += freq_shift_mono(d_src, shift, 1.0/sr) * np.exp(-(i+1)*p["decay"])

        if progress_cb:
            progress_cb(int((i+1)*100/p["m"]))

    b,a = signal.butter(p["filter_order"], p["post_hp"]/(sr/2),'highpass')
    d_res = signal.filtfilt(b,a,d_res)

    return x + d_res

# ===== Worker =====
class Worker(QtCore.QThread):
    sig_file = QtCore.Signal(int)   # ファイル内進捗
    sig_all = QtCore.Signal(int)    # 全体進捗

    def __init__(self, files):
        super().__init__()
        self.files = files

    def run(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        total = len(self.files)

        for i, f in enumerate(self.files, 1):

            y, sr = librosa.load(f, mono=False, sr=None)

            if y.ndim == 1:
                y = y[np.newaxis, :]

            if sr != PARAMS["target_sr"]:
                y = resampy.resample(y, sr, PARAMS["target_sr"])
                sr = PARAMS["target_sr"]

            def step_cb(pct):
                self.sig_file.emit(pct)

            y_out = zansei_impl(y, sr, step_cb)

            base = os.path.basename(f)
            out = os.path.join(OUTPUT_DIR, base)

            save_out(f, y_out, sr, out)

            self.sig_all.emit(int(i*100/total))
            self.sig_file.emit(100)

# ===== UI =====
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("DSRE")
        self.resize(700,500)

        self.list_files = QtWidgets.QListWidget()

        self.pb_file = QtWidgets.QProgressBar()
        self.pb_all = QtWidgets.QProgressBar()

        self.btn_start = QtWidgets.QPushButton("処理開始")

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel("対象ファイル"))
        layout.addWidget(self.list_files)
        layout.addWidget(QtWidgets.QLabel("現在ファイル進捗"))
        layout.addWidget(self.pb_file)
        layout.addWidget(QtWidgets.QLabel("全体進捗"))
        layout.addWidget(self.pb_all)
        layout.addWidget(self.btn_start)

        self.setLayout(layout)

        self.btn_start.clicked.connect(self.start)

        self.load_files()

    # ★ Outputにあるものは除外
    def load_files(self):
        if not os.path.exists(INPUT_DIR):
            return

        existing = set()
        if os.path.exists(OUTPUT_DIR):
            existing = {os.path.splitext(f)[0] for f in os.listdir(OUTPUT_DIR)}

        for f in os.listdir(INPUT_DIR):
            if not f.lower().endswith(".flac"):
                continue

            name = os.path.splitext(f)[0]
            if name in existing:
                continue

            self.list_files.addItem(os.path.join(INPUT_DIR, f))

    def start(self):
        files = [self.list_files.item(i).text() for i in range(self.list_files.count())]

        self.worker = Worker(files)
        self.worker.sig_file.connect(self.pb_file.setValue)
        self.worker.sig_all.connect(self.pb_all.setValue)
        self.worker.start()

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
