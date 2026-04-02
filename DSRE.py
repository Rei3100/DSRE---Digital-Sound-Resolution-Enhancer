import os
import sys
import traceback
from typing import Optional

import subprocess
import soundfile as sf
import tempfile

import numpy as np
from scipy import signal
import librosa
import resampy

from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QIcon, QTextCursor


# ===== 固定値 =====
FIXED_OUTDIR = r"C:\Audio\DSRE\Output"
FIXED_INPUTDIR = r"C:\Audio\DSRE"
FIXED_FORMAT = "FLAC"


def add_ffmpeg_to_path():
    if hasattr(sys, "_MEIPASS"):
        ffmpeg_dir = os.path.join(sys._MEIPASS, "ffmpeg")
    else:
        ffmpeg_dir = os.path.join(os.path.dirname(__file__), "ffmpeg")
    os.environ["PATH"] += os.pathsep + ffmpeg_dir

add_ffmpeg_to_path()


# ===== save（変更：FLAC固定 + 非表示起動）=====
def save_wav24_out(in_path, y_out, sr, out_path, fmt="FLAC", normalize=True):
    import tempfile, subprocess, numpy as np, soundfile as sf, os

    if y_out.ndim == 1:
        data = y_out[:, None]
    else:
        data = y_out.T if y_out.shape[0] < y_out.shape[1] else y_out

    data = data.astype(np.float32, copy=False)
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

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW
    )

    os.remove(tmp_wav.name)
    return out_path


# ===== DSP（変更なし）=====
def freq_shift_mono(x, f_shift, d_sr):
    N = len(x)
    S = signal.hilbert(x)
    return (S * np.exp(2j * np.pi * f_shift * d_sr * np.arange(N))).real


def zansei_impl(x, sr, params):
    b, a = signal.butter(params["filter_order"], params["pre_hp"] / (sr / 2), 'highpass')
    d_src = signal.filtfilt(b, a, x)

    d_res = np.zeros_like(x)

    for i in range(params["m"]):
        shift = sr * (i + 1) / (params["m"] * 2.0)
        d_res += freq_shift_mono(d_src, shift, 1.0 / sr) * np.exp(-(i + 1) * params["decay"])

    b, a = signal.butter(params["filter_order"], params["post_hp"] / (sr / 2), 'highpass')
    d_res = signal.filtfilt(b, a, d_res)

    return x + d_res


# ===== Worker =====
class DSREWorker(QtCore.QThread):
    sig_overall_progress = QtCore.Signal(int, int)

    def __init__(self, files, params, parent=None):
        super().__init__(parent)
        self.files = files
        self.params = params

    def run(self):
        os.makedirs(FIXED_OUTDIR, exist_ok=True)

        total = len(self.files)

        for i, in_path in enumerate(self.files, 1):
            y, sr = librosa.load(in_path, mono=True, sr=None)

            if sr != self.params["target_sr"]:
                y = resampy.resample(y, sr, self.params["target_sr"])
                sr = self.params["target_sr"]

            y_out = zansei_impl(y, sr, self.params)

            base = os.path.basename(in_path)
            out_path = os.path.join(FIXED_OUTDIR, base)

            save_wav24_out(in_path, y_out, sr, out_path)

            self.sig_overall_progress.emit(i, total)


# ===== GUI =====
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("DSRE")
        self.resize(900, 600)

        # 入力ファイル
        self.list_files = QtWidgets.QListWidget()
        self.btn_add = QtWidgets.QPushButton("入力ファイルを追加")
        self.btn_clear = QtWidgets.QPushButton("一覧をクリア")

        # 出力先（固定表示）
        self.le_outdir = QtWidgets.QLineEdit(FIXED_OUTDIR)
        self.le_outdir.setReadOnly(True)

        # パラメータ（固定）
        self.sb_m = QtWidgets.QSpinBox()
        self.sb_m.setValue(8)
        self.sb_m.setEnabled(False)

        self.dsb_decay = QtWidgets.QDoubleSpinBox()
        self.dsb_decay.setValue(1.25)
        self.dsb_decay.setEnabled(False)

        self.sb_pre = QtWidgets.QSpinBox()
        self.sb_pre.setValue(3000)
        self.sb_pre.setEnabled(False)

        self.sb_post = QtWidgets.QSpinBox()
        self.sb_post.setValue(16000)
        self.sb_post.setEnabled(False)

        self.sb_order = QtWidgets.QSpinBox()
        self.sb_order.setValue(11)
        self.sb_order.setEnabled(False)

        self.sb_sr = QtWidgets.QSpinBox()
        self.sb_sr.setValue(96000)
        self.sb_sr.setEnabled(False)

        # 進捗
        self.pb_all = QtWidgets.QProgressBar()

        # ボタン
        self.btn_start = QtWidgets.QPushButton("処理開始")

        # レイアウト
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel("入力ファイル"))
        layout.addWidget(self.list_files)
        layout.addWidget(self.btn_add)
        layout.addWidget(self.btn_clear)

        layout.addWidget(QtWidgets.QLabel("出力フォルダ"))
        layout.addWidget(self.le_outdir)

        layout.addWidget(QtWidgets.QLabel("パラメータ（固定）"))
        layout.addWidget(self.sb_m)
        layout.addWidget(self.dsb_decay)
        layout.addWidget(self.sb_pre)
        layout.addWidget(self.sb_post)
        layout.addWidget(self.sb_order)
        layout.addWidget(self.sb_sr)

        layout.addWidget(self.pb_all)
        layout.addWidget(self.btn_start)

        self.setLayout(layout)

        self.btn_add.clicked.connect(self.on_add_files)
        self.btn_clear.clicked.connect(self.list_files.clear)
        self.btn_start.clicked.connect(self.on_start)

    def on_add_files(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "ファイル選択",
            FIXED_INPUTDIR,
            "音声ファイル (*.wav *.mp3 *.flac *.m4a)"
        )
        for f in files:
            self.list_files.addItem(f)

    def params(self):
        return dict(
            m=8,
            decay=1.25,
            pre_hp=3000,
            post_hp=16000,
            target_sr=96000,
            filter_order=11,
        )

    def on_start(self):
        files = [self.list_files.item(i).text() for i in range(self.list_files.count())]

        self.worker = DSREWorker(files, self.params())
        self.worker.sig_overall_progress.connect(self.update_progress)
        self.worker.start()

    def update_progress(self, cur, total):
        self.pb_all.setValue(int(cur * 100 / total))


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
