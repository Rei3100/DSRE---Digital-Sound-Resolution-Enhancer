# ===== 強化版（処理維持＋安定性＋精度向上）=====

import os
import sys
import time
import tempfile
import subprocess
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from scipy import signal
import librosa
import resampy

from PySide6 import QtCore, QtWidgets

from send2trash import send2trash


# ===== 入出力 =====
INPUT_DIR = r"C:\Audio\DSRE"
OUTPUT_DIR = r"C:\Audio\DSRE\Output"


# ===== DSP パラメータ =====
HARMONIC_LAYERS = 8         # 倍音重畳の段数
HARMONIC_DECAY = 1.25       # 各段の減衰係数
PRE_HP_CUTOFF_HZ = 3000     # 倍音抽出前のハイパス
POST_HP_CUTOFF_HZ = 16000   # 倍音生成後のハイパス
TARGET_SR = 96000           # リサンプル先サンプリング周波数
FILTER_ORDER = 11           # バターワース次数


@dataclass(frozen=True)
class DSREParams:
    m: int = HARMONIC_LAYERS
    decay: float = HARMONIC_DECAY
    pre_hp: int = PRE_HP_CUTOFF_HZ
    post_hp: int = POST_HP_CUTOFF_HZ
    target_sr: int = TARGET_SR
    filter_order: int = FILTER_ORDER
    format: str = "FLAC"


PARAMS = DSREParams()


# ===== ffmpeg PATH 補完（同梱 ffmpeg/ffmpeg.exe があれば追加）=====
def add_ffmpeg_to_path() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    bundled = os.path.join(here, "ffmpeg", "ffmpeg.exe")
    if os.path.isfile(bundled):
        os.environ["PATH"] = os.path.dirname(bundled) + os.pathsep + os.environ.get("PATH", "")


# ===== subprocess 起動（コマンドプロンプト非表示）=====
def run_hidden(cmd):
    return subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


# ===== 安全読み込み =====
def load_audio_safe(path):
    try:
        data, sr = sf.read(path, always_2d=True)
        return data.T, sr
    except (RuntimeError, OSError, ValueError):
        pass
    try:
        y, sr = librosa.load(path, mono=False, sr=None)
        if y.ndim == 1:
            y = y[np.newaxis, :]
        return y, sr
    except Exception as e:
        raise RuntimeError(f"読み込み失敗: {path}") from e


# ===== 保存 =====
def save_wav24_out(in_path, y_out, sr, out_path):
    if y_out.ndim == 1:
        data = y_out.reshape(-1, 1)
    else:
        data = y_out.T

    data = data.astype(np.float32, copy=False)

    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 1.0:
        data /= peak

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    try:
        sf.write(tmp.name, data, sr, subtype="FLOAT")

        out_path = os.path.splitext(out_path)[0] + ".flac"
        cmd = [
            "ffmpeg", "-y",
            "-i", tmp.name,
            "-i", in_path,
            "-map", "0:a",
            "-map_metadata", "1",
            "-c:a", "flac",
            "-sample_fmt", "s32",
            out_path,
        ]
        run_hidden(cmd)
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass
    return out_path


# ===== DSP =====
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
    b, a = safe_butter(PARAMS.filter_order, PARAMS.pre_hp, sr)
    d_src = signal.filtfilt(b, a, x)

    d_sr = 1.0 / sr
    f_dn = freq_shift_mono if (x.ndim == 1) else freq_shift_multi
    d_res = np.zeros_like(x)

    total = PARAMS.m

    for i in range(total):
        if abort_cb and abort_cb():
            break

        shift = sr * (i + 1) / (total * 2.0)
        d_res += f_dn(d_src, shift, d_sr) * np.exp(-(i + 1) * PARAMS.decay)

        if progress_cb:
            progress_cb(i + 1, total)

    b, a = safe_butter(PARAMS.filter_order, PARAMS.post_hp, sr)
    d_res = signal.filtfilt(b, a, d_res)

    adp = float(np.mean(np.abs(d_res)))
    src = float(np.mean(np.abs(x)))

    eps = 1e-12
    adj = src / (adp + src + eps)

    result = (x + d_res) * adj
    if not np.all(np.isfinite(result)):
        return np.clip(x, -1.0, 1.0)
    return result


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
        self._mutex = QtCore.QMutex()
        self._wait = QtCore.QWaitCondition()
        self._failed: list[str] = []
        self._trash_failed = 0

    def abort(self):
        self._mutex.lock()
        self._abort = True
        self._wait.wakeAll()
        self._mutex.unlock()

    def pause_toggle(self):
        self._mutex.lock()
        self._pause = not self._pause
        if not self._pause:
            self._wait.wakeAll()
        self._mutex.unlock()

    def _wait_if_paused(self):
        self._mutex.lock()
        while self._pause and not self._abort:
            self._wait.wait(self._mutex)
        self._mutex.unlock()

    def run(self):
        total = len(self.files)
        start = time.time()
        succeeded = 0

        for idx, path in enumerate(self.files, 1):
            if self._abort:
                break

            self._wait_if_paused()
            if self._abort:
                break

            self.sig_text.emit(f"{idx}/{total}")

            try:
                y, sr = load_audio_safe(path)

                if sr != PARAMS.target_sr:
                    y = resampy.resample(y, sr, PARAMS.target_sr)
                    sr = PARAMS.target_sr

                def step_cb(cur, m):
                    self.sig_step.emit(int(cur * 100 / m))

                y_out = zansei_impl(
                    y, sr,
                    progress_cb=step_cb,
                    abort_cb=lambda: self._abort,
                )

                out = os.path.join(OUTPUT_DIR, os.path.basename(path))
                save_wav24_out(path, y_out, sr, out)

                try:
                    send2trash(path)
                except Exception:
                    self._trash_failed += 1

                succeeded += 1

            except Exception:
                self._failed.append(os.path.basename(path))

            elapsed = time.time() - start
            remain = (elapsed / idx) * (total - idx)
            fail_n = len(self._failed)
            parts = []
            if fail_n:
                parts.append(f"失敗{fail_n}")
            if self._trash_failed:
                parts.append(f"ゴミ箱{self._trash_failed}")
            suffix = ("  " + "  ".join(parts)) if parts else ""
            self.sig_text.emit(f"{idx}/{total}  残り{int(remain)}秒{suffix}")

            self.sig_all.emit(int(idx * 100 / total))
            self.sig_step.emit(100)

        fail_n = len(self._failed)
        trash_n = self._trash_failed
        extras = []
        if fail_n:
            extras.append(f"失敗{fail_n}")
        if trash_n:
            extras.append(f"ゴミ箱{trash_n}")
        tail = ("  " + "  ".join(extras)) if extras else ""
        if self._abort:
            self.sig_text.emit(f"中断  成功{succeeded}/{total}{tail}")
        elif extras:
            self.sig_text.emit(f"完了  成功{succeeded}/{total}{tail}")
        else:
            self.sig_text.emit(f"完了  {succeeded}/{total}")


# ===== UI（表示・レイアウトは変更なし、動作のみ堅牢化）=====
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("DSRE")
        self.resize(320, 180)

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

        if not os.path.exists(INPUT_DIR):
            return files

        for f in os.listdir(INPUT_DIR):
            if f.lower().endswith(".flac"):
                if os.path.splitext(f)[0] not in existing:
                    files.append(os.path.join(INPUT_DIR, f))

        return files

    def start(self):
        if self.worker and self.worker.isRunning():
            return
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

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            ret = QtWidgets.QMessageBox.question(
                self, "DSRE", "処理中です。終了しますか？",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if ret != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
            self.worker.abort()
            self.worker.wait(3000)
        event.accept()


def _run_selftest() -> int:
    """ビルド成果物が最低限 import できることを確認するセルフテスト。

    PyInstaller の excludes や hidden import の抜けで起動不能になる事故を
    CI / ローカル / デプロイ直前で捕まえるためのゲート。QApplication は作らない
    (ヘッドレス CI 環境で Qt platform plugin 初期化を走らせないため)。
    """
    import traceback
    log_dir = os.path.dirname(sys.executable) or os.getcwd()
    log_path = os.path.join(log_dir, "selftest.log")
    try:
        import numpy as _np
        import scipy as _sp
        import scipy.signal as _sps
        import scipy.linalg  # noqa: F401
        import scipy.fft  # noqa: F401
        import numpy.testing  # noqa: F401  # unittest 地雷検出用
        import librosa as _lb
        import resampy  # noqa: F401
        import soundfile  # noqa: F401
        import send2trash  # noqa: F401
        from PySide6 import QtCore, QtWidgets  # noqa: F401
        _ = _sps.butter
        _ = _sps.filtfilt
        _ = _sps.hilbert
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(
                    f"selftest OK numpy={_np.__version__} "
                    f"scipy={_sp.__version__} librosa={_lb.__version__}\n"
                )
        except Exception:
            pass
        return 0
    except Exception:
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("selftest FAILED\n")
                traceback.print_exc(file=f)
        except Exception:
            traceback.print_exc()
        return 1


def main():
    if "--selftest" in sys.argv:
        sys.exit(_run_selftest())
    add_ffmpeg_to_path()
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
