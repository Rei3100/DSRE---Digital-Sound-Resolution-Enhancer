# ==== 全置き換え ====

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

# ===== 固定設定 =====
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

# ===== ffmpegパス =====
def add_ffmpeg_to_path():
    if hasattr(sys, "_MEIPASS"):
        ffmpeg_dir = os.path.join(sys._MEIPASS, "ffmpeg")
    else:
        ffmpeg_dir = os.path.join(os.path.dirname(__file__), "ffmpeg")
    os.environ["PATH"] += os.pathsep + ffmpeg_dir

add_ffmpeg_to_path()

# ===== 保存 =====
def save_out(in_path, y_out, sr, out_path):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()

    sf.write(tmp.name, y_out, sr, subtype="FLOAT")

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
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW
    )

    os.remove(tmp.name)
    return out_path

# ===== DSP =====
def freq_shift(x, sr):
    p = PARAMS
    b,a = signal.butter(p["filter_order"], p["pre_hp"]/(sr/2),'highpass')
    d = signal.filtfilt(b,a,x)

    res = np.zeros_like(x)

    for i in range(p["m"]):
        shift = sr*(i+1)/(p["m"]*2.0)
        res += signal.hilbert(d).real * np.exp(-(i+1)*p["decay"])

    b,a = signal.butter(p["filter_order"], p["post_hp"]/(sr/2),'highpass')
    res = signal.filtfilt(b,a,res)

    return x + res

# ===== Worker =====
class Worker(QtCore.QThread):
    sig_progress = QtCore.Signal(int)

    def __init__(self, files):
        super().__init__()
        self.files = files

    def run(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        total = len(self.files)

        for i, f in enumerate(self.files, 1):

            base = os.path.basename(f)
            out = os.path.join(OUTPUT_DIR, base)

            # ★ 既に出力がある場合スキップ
            if os.path.exists(os.path.splitext(out)[0] + ".flac"):
                self.sig_progress.emit(int(i*100/total))
                continue

            y, sr = librosa.load(f, mono=True, sr=None)

            if sr != PARAMS["target_sr"]:
                y = resampy.resample(y, sr, PARAMS["target_sr"])
                sr = PARAMS["target_sr"]

            y_out = freq_shift(y, sr)

            save_out(f, y_out, sr, out)

            self.sig_progress.emit(int(i*100/total))

# ===== UI =====
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("DSRE")
        self.resize(600,400)

        self.list_files = QtWidgets.QListWidget()

        self.btn_add = QtWidgets.QPushButton("ファイル追加")
        self.btn_start = QtWidgets.QPushButton("開始")

        self.pb = QtWidgets.QProgressBar()

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel("入力ファイル"))
        layout.addWidget(self.list_files)
        layout.addWidget(self.btn_add)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.pb)

        self.setLayout(layout)

        self.btn_add.clicked.connect(self.add_files)
        self.btn_start.clicked.connect(self.start)

        # ★ 起動時自動読み込み
        self.load_default_files()

    def load_default_files(self):
        if not os.path.exists(INPUT_DIR):
            return
        for f in os.listdir(INPUT_DIR):
            if f.lower().endswith(".flac"):
                self.list_files.addItem(os.path.join(INPUT_DIR,f))

    def add_files(self):
        files,_ = QtWidgets.QFileDialog.getOpenFileNames(
            self,"ファイル選択",INPUT_DIR,"FLAC (*.flac)"
        )
        for f in files:
            self.list_files.addItem(f)

    def start(self):
        files = [self.list_files.item(i).text() for i in range(self.list_files.count())]

        self.worker = Worker(files)
        self.worker.sig_progress.connect(self.pb.setValue)
        self.worker.start()

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
