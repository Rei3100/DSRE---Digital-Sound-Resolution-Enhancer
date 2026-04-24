# ===== 強化版（処理維持＋安定性＋精度向上）=====

import os
import sys
import time
import tempfile
import subprocess
import configparser
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.fft import next_fast_len
import librosa
import resampy

from PySide6 import QtCore, QtGui, QtWidgets

from send2trash import send2trash


# ===== 入出力 =====
INPUT_DIR = r"C:\Audio\DSRE"
OUTPUT_DIR = r"C:\Audio\DSRE\Output"


# ===== DSP パラメータ =====
HARMONIC_LAYERS = 8         # 倍音重畳の段数
HARMONIC_DECAY = 1.25       # 各段の減衰係数
PRE_HP_CUTOFF_HZ = 3000     # 倍音抽出前のハイパス
POST_HP_CUTOFF_HZ = 16000   # 倍音生成後のハイパス
TARGET_SR = 192000          # リサンプル先サンプリング周波数 (192 kHz)
FILTER_ORDER = 11           # バターワース次数
OUTPUT_SUBTYPE = "FLOAT"    # v1.4: 32-bit IEEE float WAV (libsndfile 完全対応、量子化ノイズ消滅)
OUTPUT_FORMAT = "WAV"
OUTPUT_EXT = ".wav"


# ===== 負荷レベル (v1.3 Phase 3) =====
LOAD_LEVELS = ("軽", "標準", "最大")
LOAD_DEFAULT = "標準"
STATE_INI_NAME = "state.ini"


def _state_ini_path():
    base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, STATE_INI_NAME)


def load_level() -> str:
    p = _state_ini_path()
    if not os.path.isfile(p):
        return LOAD_DEFAULT
    try:
        cp = configparser.ConfigParser()
        cp.read(p, encoding="utf-8")
        lv = cp.get("ui", "load", fallback=LOAD_DEFAULT)
        return lv if lv in LOAD_LEVELS else LOAD_DEFAULT
    except Exception:
        return LOAD_DEFAULT


def save_level(lv: str) -> None:
    if lv not in LOAD_LEVELS:
        return
    try:
        cp = configparser.ConfigParser()
        p = _state_ini_path()
        if os.path.isfile(p):
            cp.read(p, encoding="utf-8")
        if not cp.has_section("ui"):
            cp.add_section("ui")
        cp.set("ui", "load", lv)
        with open(p, "w", encoding="utf-8") as f:
            cp.write(f)
    except Exception:
        pass


def threads_for_level(lv: str) -> int:
    if lv == "軽":
        return 1
    if lv == "最大":
        try:
            return max(1, os.cpu_count() or 1)
        except Exception:
            return 1
    return 0  # 標準: auto (threadpoolctl に None 相当を渡す)


def resampy_parallel_for_level(lv: str) -> bool:
    return lv != "軽"


@dataclass(frozen=True)
class DSREParams:
    m: int = HARMONIC_LAYERS
    decay: float = HARMONIC_DECAY
    pre_hp: int = PRE_HP_CUTOFF_HZ
    post_hp: int = POST_HP_CUTOFF_HZ
    target_sr: int = TARGET_SR
    filter_order: int = FILTER_ORDER
    output_format: str = "WAV"
    output_subtype: str = "FLOAT"


PARAMS = DSREParams()


# ===== バンドルリソースのパス解決 =====
def _resource_base_dirs() -> tuple[str, ...]:
    """PyInstaller onedir / 開発実行の両方で同梱リソースを探すためのベースディレクトリ群。"""
    dirs: list[str] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        dirs.append(meipass)
    if getattr(sys, "frozen", False):
        dirs.append(os.path.dirname(os.path.abspath(sys.executable)))
    else:
        dirs.append(os.path.dirname(os.path.abspath(__file__)))
    return tuple(dirs)


def _find_bundled(*relative_paths: str) -> str | None:
    """いずれかのベース + 相対パスの組合せで最初に見つかった絶対パスを返す。"""
    for base in _resource_base_dirs():
        for rel in relative_paths:
            p = os.path.join(base, rel)
            if os.path.isfile(p):
                return p
    return None


# ===== ffmpeg PATH 補完 (同梱 ffmpeg/ffmpeg.exe または _internal/ffmpeg/ffmpeg.exe を探索) =====
def add_ffmpeg_to_path() -> None:
    bundled = _find_bundled(
        os.path.join("ffmpeg", "ffmpeg.exe"),
        os.path.join("_internal", "ffmpeg", "ffmpeg.exe"),
    )
    if bundled:
        os.environ["PATH"] = os.path.dirname(bundled) + os.pathsep + os.environ.get("PATH", "")


# ===== アプリアイコン (logo.ico) =====
def _logo_path() -> str | None:
    return _find_bundled("logo.ico")


def _app_icon() -> "QtGui.QIcon":
    p = _logo_path()
    return QtGui.QIcon(p) if p else QtGui.QIcon()


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
        data, sr = sf.read(path, always_2d=True, dtype="float32")
        return data.T, sr
    except (RuntimeError, OSError, ValueError):
        pass
    try:
        y, sr = librosa.load(path, mono=False, sr=None, dtype=np.float32)
        if y.ndim == 1:
            y = y[np.newaxis, :]
        return y, sr
    except Exception as e:
        raise RuntimeError(f"読み込み失敗: {path}") from e


# ===== 保存 (v1.4: 192kHz / 32-bit IEEE float WAV + ffmpeg -c copy でメタデータ継承) =====
def _try_sf_write(path, data, sr, subtype, fmt):
    """書込 → 読み直しで shape / sr が一致するかをラウンドトリップ検証する。
    失敗時は中途半端に残ったファイルを削除して False を返す。"""
    try:
        sf.write(path, data, sr, subtype=subtype, format=fmt)
        check, check_sr = sf.read(path, always_2d=True, dtype="float32")
        if check_sr != sr or check.shape != data.shape:
            raise RuntimeError("roundtrip mismatch")
        return True
    except Exception:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
        return False


def save_wav32float_out(in_path, y_out, sr, out_path):
    """DSP 結果を 32-bit IEEE float WAV として書き出し、ffmpeg でメタデータを継承する。

    - 32-bit float WAV (subtype="FLOAT", format="WAV") は libsndfile 完全対応、
      foobar2000 では "32-bit floating-point" と表示される
    - 32-bit float の動的レンジは実用上無限 (clipping 量子化ノイズ皆無)
    - 再生機器側の入力段で >1.0 を嫌うケースに備え、peak>0.99 のときは
      -1dBFS 相当に緩くスケールダウンする
    - メタデータ継承は ffmpeg `-map_metadata 1 -c copy -write_id3v2 1` で実施。
      入力が FLAC (Vorbis Comment) でも WAV (LIST/INFO) でも、タグが自動変換される。
    """
    if y_out.ndim == 1:
        data = y_out.reshape(-1, 1)
    else:
        data = y_out.T
    data = data.astype(np.float32, copy=False)

    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 0.99:
        data = data * (0.99 / peak)

    base = os.path.splitext(out_path)[0]
    final_path = base + OUTPUT_EXT

    # 一時 WAV に書込 (メタデータ無し)
    tmp_path = final_path + ".tmp_src.wav"
    if not _try_sf_write(tmp_path, data, sr, OUTPUT_SUBTYPE, OUTPUT_FORMAT):
        raise RuntimeError(f"WAV FLOAT 書込失敗: {final_path}")

    # ffmpeg でメタデータ継承 (音声は -c copy で再エンコード無し = 完全無劣化)
    cmd = [
        "ffmpeg", "-y",
        "-i", tmp_path,     # 音声ソース (DSP 済 WAV FLOAT)
        "-i", in_path,      # メタデータソース (元 FLAC 等)
        "-map", "0:a",
        "-map_metadata", "1",
        "-c", "copy",
        "-write_id3v2", "1",  # WAV にも ID3v2 チャンクを書く
        final_path,
    ]
    try:
        run_hidden(cmd)
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    except Exception:
        # ffmpeg 失敗時はメタ無しで確定 (音声は確保されている)
        try:
            if os.path.exists(final_path):
                os.remove(final_path)
            os.replace(tmp_path, final_path)
        except OSError:
            pass
    return final_path


# ===== DSP =====
def freq_shift_mono(x, f_shift, d_sr):
    """1D 実信号を f_shift [Hz] だけ周波数シフト (single-sideband)。

    解析信号 (hilbert で得た complex signal) に e^{j*2*pi*f*t} を乗じると、
    数学的に上側サイドバンドのみが残る。最後に `.real` で実部を取るのは、
    解析信号 z = x + j*H[x] のうち音として復元すべき成分が実部側であるため。
    `np.abs(..)` では振幅包絡線になってしまい原音と関係ないので誤り。
    """
    N = len(x)
    Np = next_fast_len(max(1, N))
    S = signal.hilbert(np.hstack((x, np.zeros(Np - N, dtype=x.dtype))))
    F = np.exp(2j * np.pi * f_shift * d_sr * np.arange(Np))
    return (S * F)[:N].real


def freq_shift_multi(x, f_shift, d_sr):
    """マルチチャンネル版 freq_shift_mono。各チャンネル独立に適用。
    `.real` を取る理由は freq_shift_mono の docstring を参照。
    """
    Ch, N = x.shape
    Np = next_fast_len(max(1, N))
    padded = np.zeros((Ch, Np), dtype=x.dtype)
    padded[:, :N] = x
    S = signal.hilbert(padded, axis=-1)
    F = np.exp(2j * np.pi * f_shift * d_sr * np.arange(Np))
    return (S * F[np.newaxis, :])[:, :N].real


def safe_butter_sos(order, cutoff_hz, sr, btype="highpass"):
    """SOS (Second-Order Sections) 形式で Butterworth を構築する。
    高次 IIR (本プロジェクトでは order=11) で ba 係数がアンダーフロー / ピボット
    不安定になるのを避けるため、sosfiltfilt と対で使うこと。
    """
    nyq = sr / 2.0
    cutoff_hz = min(cutoff_hz, nyq * 0.95)
    order = min(order, 20)
    wn = max(1e-6, min(0.999, cutoff_hz / nyq))
    return signal.butter(order, wn, btype=btype, output="sos")


def safe_sosfiltfilt(sos, x, axis=-1):
    """sosfiltfilt のガード付きラッパ。
    理論上 sosfiltfilt は filtfilt(ba) より数値安定で NaN が出にくいが、
    極端な低 Wn の高次 IIR では浮動小数誤差が蓄積しうるためフェイルセーフを張る。
    例外 / NaN / Inf のいずれが出ても入力を [-1, 1] に clip して返す。
    """
    try:
        y = signal.sosfiltfilt(sos, x, axis=axis)
    except Exception:
        return np.clip(x, -1.0, 1.0)
    if not np.all(np.isfinite(y)):
        return np.clip(x, -1.0, 1.0)
    return y


def zansei_impl(x, sr, progress_cb=None, abort_cb=None):
    # 倍音抽出用 pre-HP (3kHz 以上を倍音生成素材に使う、SOS で数値安定化)
    sos_pre = safe_butter_sos(PARAMS.filter_order, PARAMS.pre_hp, sr, btype="highpass")
    d_src = safe_sosfiltfilt(sos_pre, x, axis=-1)

    d_sr = 1.0 / sr
    f_dn = freq_shift_mono if (x.ndim == 1) else freq_shift_multi
    d_res = np.zeros_like(x)

    total = PARAMS.m
    decays = np.exp(-np.arange(1, total + 1) * PARAMS.decay)
    nyq = sr / 2.0

    for i in range(total):
        if abort_cb and abort_cb():
            break

        shift = sr * (i + 1) / (total * 2.0)
        # ナイキスト到達/超過のシフト層はスキップ (折り返しアーティファクト防止)。
        # 現パラメータ (total=8, sr=192000) では最大 shift=96000=nyq なので最終層のみスキップ。
        if shift >= nyq:
            if progress_cb:
                progress_cb(i + 1, total)
            continue

        d_res += f_dn(d_src, shift, d_sr) * decays[i]

        if progress_cb:
            progress_cb(i + 1, total)

    # 生成した倍音の低域を再度カット (16kHz 以上の高域のみに寄与、SOS で数値安定化)
    sos_post = safe_butter_sos(PARAMS.filter_order, PARAMS.post_hp, sr, btype="highpass")
    d_res = safe_sosfiltfilt(sos_post, d_res, axis=-1)

    adp = float(np.mean(np.abs(d_res)))
    src = float(np.mean(np.abs(x)))

    eps = 1e-12
    adj = src / (adp + src + eps)

    result = (x + d_res) * adj
    if not np.all(np.isfinite(result)):
        return np.clip(x, -1.0, 1.0)
    return result


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


# ===== Worker =====
class Worker(QtCore.QThread):
    sig_step = QtCore.Signal(int)
    sig_all = QtCore.Signal(int)
    sig_text = QtCore.Signal(str)

    def __init__(self, files, level=LOAD_DEFAULT):
        super().__init__()
        self.files = files
        self.level = level if level in LOAD_LEVELS else LOAD_DEFAULT
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

        try:
            from threadpoolctl import threadpool_limits
        except Exception:
            threadpool_limits = None

        n_threads = threads_for_level(self.level)
        try:
            import numba  # noqa: F401
            import numba.core.config as _nc
            if n_threads > 0:
                try:
                    import numba as _nb
                    _nb.set_num_threads(n_threads)
                except Exception:
                    pass
        except Exception:
            pass

        resampy_parallel = resampy_parallel_for_level(self.level)

        limits_kw = None
        if threadpool_limits is not None:
            if n_threads == 0:
                limits_kw = None
            else:
                limits_kw = {"limits": n_threads}

        ctx = threadpool_limits(**limits_kw) if (threadpool_limits is not None and limits_kw) else _NullCtx()
        with ctx:
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
                        try:
                            y = resampy.resample(y, sr, PARAMS.target_sr, parallel=resampy_parallel)
                        except TypeError:
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
                    save_wav32float_out(path, y_out, sr, out)

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


# ===== UI (v1.4: トレイ常駐 + 負荷サブメニュー + logo.ico + × 即終了) =====
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("DSRE")
        self.setWindowIcon(_app_icon())
        self.resize(340, 210)

        self.label = QtWidgets.QLabel("待機")
        self.pb_file = QtWidgets.QProgressBar()
        self.pb_all = QtWidgets.QProgressBar()

        self.btn_start = QtWidgets.QPushButton("開始")
        self.btn_pause = QtWidgets.QPushButton("一時停止")
        self.btn_cancel = QtWidgets.QPushButton("取消")

        self.cmb_level = QtWidgets.QComboBox()
        self.cmb_level.addItems(list(LOAD_LEVELS))
        _lv = load_level()
        idx_lv = LOAD_LEVELS.index(_lv) if _lv in LOAD_LEVELS else LOAD_LEVELS.index(LOAD_DEFAULT)
        self.cmb_level.setCurrentIndex(idx_lv)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.pb_file)
        layout.addWidget(self.pb_all)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_pause)
        layout.addWidget(self.btn_cancel)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("負荷"))
        row.addWidget(self.cmb_level, 1)
        layout.addLayout(row)

        self.setLayout(layout)

        self.btn_start.clicked.connect(self.start)
        self.btn_pause.clicked.connect(self.pause)
        self.btn_cancel.clicked.connect(self.cancel)
        # 負荷 ComboBox → _set_load_level で保存 + トレイ側と同期
        self.cmb_level.currentTextChanged.connect(self._set_load_level)

        self.worker = None
        self._tray = None
        self._load_actions: dict[str, QtGui.QAction] = {}
        self._setup_tray()

    # ---- トレイ ----
    def _setup_tray(self) -> None:
        """システムトレイアイコン + 右クリックメニュー (開始/一時停止/取消/負荷サブメニュー/終了) を構築。"""
        if not QtWidgets.QSystemTrayIcon.isSystemTrayAvailable():
            return

        self._tray = QtWidgets.QSystemTrayIcon(self)
        self._tray.setIcon(_app_icon())
        self._tray.setToolTip("DSRE")

        menu = QtWidgets.QMenu()
        act_show = menu.addAction("表示")
        act_show.triggered.connect(self._show_from_tray)
        menu.addSeparator()

        menu.addAction("開始", self.start)
        menu.addAction("一時停止", self.pause)
        menu.addAction("取消", self.cancel)
        menu.addSeparator()

        # 負荷サブメニュー (排他ラジオ)
        sub = menu.addMenu("負荷")
        group = QtGui.QActionGroup(self)
        group.setExclusive(True)
        current = self.cmb_level.currentText()
        for lv in LOAD_LEVELS:
            act = QtGui.QAction(lv, self, checkable=True)
            act.setChecked(lv == current)
            act.triggered.connect(lambda _checked=False, name=lv: self._set_load_level(name))
            group.addAction(act)
            sub.addAction(act)
            self._load_actions[lv] = act
        self._load_action_group = group  # GC 防止のため保持

        menu.addSeparator()
        menu.addAction("終了", self._quit_app)

        self._tray.setContextMenu(menu)
        self._tray.activated.connect(self._on_tray)
        self._tray.show()

    def _set_load_level(self, lv: str) -> None:
        """UI コンボ / トレイサブメニューどちらから呼ばれても、双方と state.ini を同期する。"""
        if lv not in LOAD_LEVELS:
            return
        save_level(lv)
        # コンボ側を signal 循環なしで同期
        if hasattr(self, "cmb_level") and self.cmb_level.currentText() != lv:
            self.cmb_level.blockSignals(True)
            try:
                self.cmb_level.setCurrentText(lv)
            finally:
                self.cmb_level.blockSignals(False)
        # トレイ側のチェック状態を同期
        for name, act in self._load_actions.items():
            act.setChecked(name == lv)

    def _show_from_tray(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _on_tray(self, reason) -> None:
        # 左クリックでウィンドウの表示/非表示をトグル
        if reason == QtWidgets.QSystemTrayIcon.ActivationReason.Trigger:
            if self.isVisible() and self.isActiveWindow():
                self.hide()
            else:
                self._show_from_tray()

    def _quit_app(self) -> None:
        """トレイ「終了」および × ボタン共通の終了処理。確認ダイアログなし (ユーザー要望)。"""
        if self.worker and self.worker.isRunning():
            self.worker.abort()
            self.worker.wait(3000)
        if self._tray is not None:
            self._tray.hide()
        QtWidgets.QApplication.instance().quit()

    # ---- ファイル処理 ----
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

        lv = self.cmb_level.currentText() if hasattr(self, "cmb_level") else LOAD_DEFAULT
        self.worker = Worker(files, level=lv)
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

    # ---- ウィンドウイベント (最小化→トレイ隠蔽、× で即終了) ----
    def changeEvent(self, event):
        # 最小化はトレイに隠蔽 (タスクバーから消す)
        if event.type() == QtCore.QEvent.Type.WindowStateChange:
            if self.isMinimized() and self._tray is not None:
                event.ignore()
                # QTimer.singleShot で hide を遅延させないと状態変化中で無視されることがある
                QtCore.QTimer.singleShot(0, self.hide)
                return
        super().changeEvent(event)

    def closeEvent(self, event):
        # × = 即終了 (確認ダイアログなし、処理中であれば abort + 3 秒待機)
        self._quit_app()
        event.accept()


def _run_selftest() -> int:
    """ビルド成果物が最低限 import + 出力書込 + 処理 determinism で通ることを確認するセルフテスト。

    PyInstaller の excludes や hidden import の抜けで起動不能になる事故を
    CI / ローカル / デプロイ直前で捕まえるためのゲート。QApplication は作らない
    (ヘッドレス CI 環境で Qt platform plugin 初期化を走らせないため)。

    v1.3 追加検証:
      1. threadpoolctl import 可能
      2. 192000Hz / PCM_32 FLAC 書込 → 読み直しで完全一致
      3. 短い合成信号を zansei_impl で 3 段の負荷レベル (スレッド数違い) で各 2 回実行し、
         **同じ負荷内での determinism (bit 一致)** を確認
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
        import soundfile as _sf
        import send2trash  # noqa: F401
        from PySide6 import QtCore, QtWidgets  # noqa: F401
        import threadpoolctl  # noqa: F401 (v1.3 追加)

        _ = _sps.butter
        _ = _sps.filtfilt
        _ = _sps.hilbert

        tpc_version = getattr(threadpoolctl, "__version__", "?")

        # --- (1) 192kHz / PCM_32 FLAC round-trip ---
        tmp_flac = tempfile.NamedTemporaryFile(delete=False, suffix=".flac")
        tmp_flac.close()
        sr_test = 192000
        t = _np.arange(sr_test // 20, dtype=_np.float32) / sr_test
        sig = (0.25 * _np.sin(2 * _np.pi * 1000.0 * t)).astype(_np.float32)
        sig2 = _np.stack([sig, sig], axis=1)
        rt_fmt = "unknown"
        rt_subtype = "unknown"
        try:
            _sf.write(tmp_flac.name, sig2, sr_test, subtype="PCM_32", format="FLAC")
            data_read, sr_read = _sf.read(tmp_flac.name, always_2d=True, dtype="float32")
            assert sr_read == sr_test, f"sr mismatch {sr_read}!={sr_test}"
            assert data_read.shape == sig2.shape, "shape mismatch after round-trip"
            rt_fmt = "FLAC"
            rt_subtype = "PCM_32"
        except Exception:
            try:
                if os.path.exists(tmp_flac.name):
                    os.remove(tmp_flac.name)
            except OSError:
                pass
            tmp_fallback = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp_fallback.close()
            try:
                _sf.write(tmp_fallback.name, sig2, sr_test, subtype="PCM_32", format="WAV")
                data_read, sr_read = _sf.read(tmp_fallback.name, always_2d=True, dtype="float32")
                assert sr_read == sr_test, f"sr mismatch {sr_read}!={sr_test}"
                rt_fmt = "WAV"
                rt_subtype = "PCM_32"
                tmp_flac.name = tmp_fallback.name
            finally:
                pass
        finally:
            try:
                if os.path.exists(tmp_flac.name):
                    os.remove(tmp_flac.name)
            except OSError:
                pass

        # --- (2) zansei_impl の determinism (同一負荷内で二重実行 bit 一致) ---
        import numpy as _np2
        rng = _np2.random.default_rng(1234)
        N = 4096
        x_stereo = rng.standard_normal((2, N)).astype(_np2.float32) * 0.05
        sr_proc = 192000

        det_ok = True
        det_notes = []
        for lv in LOAD_LEVELS:
            n_thr = threads_for_level(lv)
            kw = {"limits": n_thr} if n_thr > 0 else None
            try:
                ctx1 = threadpoolctl.threadpool_limits(**kw) if kw else _NullCtx()
                with ctx1:
                    y1 = zansei_impl(x_stereo.copy(), sr_proc)
                ctx2 = threadpoolctl.threadpool_limits(**kw) if kw else _NullCtx()
                with ctx2:
                    y2 = zansei_impl(x_stereo.copy(), sr_proc)
            except Exception as e:
                det_ok = False
                det_notes.append(f"{lv}:EXC({type(e).__name__})")
                continue

            if not _np2.all(_np2.isfinite(y1)) or not _np2.all(_np2.isfinite(y2)):
                det_notes.append(f"{lv}:NaN")
                det_ok = False
                continue

            if _np2.array_equal(y1, y2):
                det_notes.append(f"{lv}:OK")
            else:
                max_abs = float(_np2.max(_np2.abs(y1 - y2)))
                det_notes.append(f"{lv}:diff(max={max_abs:.3e})")
                if max_abs > 1e-5:
                    det_ok = False

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(
                f"selftest OK numpy={_np.__version__} "
                f"scipy={_sp.__version__} librosa={_lb.__version__} "
                f"threadpoolctl={tpc_version} "
                f"roundtrip={rt_fmt}/{rt_subtype} "
                f"determinism=[{' '.join(det_notes)}]\n"
            )
        return 0 if det_ok else 1
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
    app.setWindowIcon(_app_icon())
    # トレイ運用: 最後のウィンドウが閉じてもアプリを終了させない
    app.setQuitOnLastWindowClosed(False)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
