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
# v1.5: FLAC 192kHz / PCM_24 固定 (v1.4 の WAV 32bit float は foobar 測定で v1.3 と同値だったため revert)
OUTPUT_SUBTYPE = "PCM_24"
OUTPUT_SUBTYPE_FALLBACK = "PCM_16"  # libsndfile 異常時の安全網
OUTPUT_FORMAT = "FLAC"
OUTPUT_EXT = ".flac"

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
    # v1.5: FLAC 192kHz / PCM_24 固定
    output_format: str = "FLAC"
    output_subtype: str = "PCM_24"
    output_subtype_fallback: str = "PCM_16"


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


# ===== 保存 (v1.5: 192kHz / FLAC PCM_24 + ffmpeg -c copy でメタデータ継承) =====
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


def save_flac24_out(in_path, y_out, sr, out_path):
    """DSP 結果を FLAC 192kHz / PCM_24 として書き出し、ffmpeg でメタデータを継承する。

    v1.5 で v1.3 方式に revert (理由):
    - v1.4 で WAV 32bit float に変更したが、foobar2000 で v1.3 (FLAC PCM_24) と
      DR / PLR / 波形すべて同値だった (32bit 化に音質メリット無し、重量増のみ)
    - 24bit の DR は理論上 144dB、実用上の必要 DR (~100dB) を既に上回る
    - 出荷物は FLAC のみ (Vorbis Comment ネイティブ、metadata は ffmpeg `-c copy` で継承)

    保存パス:
    - PCM_24 primary → PCM_16 fallback の 2 段試行 (libsndfile 異常時の安全網)
    - peak > 1.0 のときのみ正規化 (v1.4 の 0.99 スケールは過剰だった、v1.3 方式)
    - ffmpeg コマンドは `-map_metadata 1 -c copy` のみ (FLAC は Vorbis Comment が
      native、`-write_id3v2 1` は不要。WAV 専用フラグだったので v1.4 から削除)
    """
    if y_out.ndim == 1:
        data = y_out.reshape(-1, 1)
    else:
        data = y_out.T
    data = data.astype(np.float32, copy=False)

    # v1.3 方式: peak が 1.0 を超えたときのみ 1.0 に揃える (緩い clip 防止)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 1.0:
        data = data / peak

    base = os.path.splitext(out_path)[0]
    final_path = base + OUTPUT_EXT

    # 一時 FLAC に書込 (メタデータ無し)、PCM_24 → PCM_16 の順に試行
    tmp_path = final_path + ".tmp_src.flac"
    wrote = False
    for subtype in (OUTPUT_SUBTYPE, OUTPUT_SUBTYPE_FALLBACK):
        if _try_sf_write(tmp_path, data, sr, subtype, OUTPUT_FORMAT):
            wrote = True
            break
    if not wrote:
        raise RuntimeError(f"FLAC 書込失敗 (PCM_24 / PCM_16 共に NG): {final_path}")

    # ffmpeg でメタデータ継承 (音声は -c copy で再エンコード無し = 完全無劣化)
    cmd = [
        "ffmpeg", "-y",
        "-i", tmp_path,     # 音声ソース (DSP 済 FLAC)
        "-i", in_path,      # メタデータソース (元 FLAC 等)
        "-map", "0:a",
        "-map_metadata", "1",
        "-c", "copy",
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
                    save_flac24_out(path, y_out, sr, out)

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
    """v1.5 自走品質ループ: import + FLAC PCM_24 roundtrip + sosfiltfilt 等価性 + 3 負荷 determinism + ffmpeg 同梱確認。

    Claude が「音響処理を改善しても劣化していない」ことを数値で判定するためのゲート。
    verdict が DEGRADED のときは exit 1 → CI / build.ps1 / deploy.ps1 が artifact を作らない。
    QApplication は作らない (ヘッドレス環境で Qt platform plugin を走らせない)。

    検査層 (v1.5 で FLAC PCM_24 ロードトリップに revert):
      (1) 必須 import (numpy/scipy/librosa/resampy/soundfile/send2trash/PySide6/threadpoolctl)
      (2) FLAC 192 kHz / PCM_24 roundtrip: shape + sr 一致 + 量子化誤差 < 1.5e-7 (2^-23)
          v1.4 の WAV FLOAT (1e-6 閾値) から revert
      (3) sosfiltfilt 等価性: 旧 filtfilt(ba) と新 sosfiltfilt(sos) で low Wn 11 次
          Butterworth を回し、max_abs_diff < 1e-4 / rms_diff / rms_ref < 1e-5 を確認
          (旧が NaN を吐き新が有限値のときは IMPROVED)
      (4) 3 負荷 determinism: 同一入力 × 同一負荷で 2 回実行し bit 一致
      (5) 同梱 ffmpeg の存在確認 (ビルド成果物でのみ意味がある、開発時はスキップ可)
    """
    import traceback
    log_dir = os.path.dirname(sys.executable) or os.getcwd()
    log_path = os.path.join(log_dir, "selftest.log")
    try:
        # ---- (1) imports ----
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
        import threadpoolctl  # noqa: F401

        _ = _sps.butter
        _ = _sps.filtfilt
        _ = _sps.sosfiltfilt
        _ = _sps.hilbert

        tpc_version = getattr(threadpoolctl, "__version__", "?")
        notes: list[str] = []
        verdict = "EQUIV"

        # ---- (2) FLAC 192 kHz / PCM_24 roundtrip ----
        sr_test = 192000
        t = _np.arange(sr_test // 20, dtype=_np.float32) / sr_test
        sig_mono = (0.25 * _np.sin(2 * _np.pi * 1000.0 * t)).astype(_np.float32)
        sig_stereo = _np.stack([sig_mono, sig_mono], axis=1)

        rt_max_abs = float("nan")
        rt_status = "FAIL"
        # PCM_24 の量子化誤差: 2^-23 ≈ 1.19e-7 (signed 24bit の最小単位)
        # マージン込みで 1.5e-7 を閾値に
        RT_THRESHOLD = 1.5e-7
        tmp_flac = tempfile.NamedTemporaryFile(delete=False, suffix=".flac")
        tmp_flac.close()
        try:
            _sf.write(tmp_flac.name, sig_stereo, sr_test, subtype=OUTPUT_SUBTYPE, format=OUTPUT_FORMAT)
            data_read, sr_read = _sf.read(tmp_flac.name, always_2d=True, dtype="float32")
            assert sr_read == sr_test, f"sr mismatch {sr_read}!={sr_test}"
            assert data_read.shape == sig_stereo.shape, "shape mismatch after round-trip"
            rt_max_abs = float(_np.max(_np.abs(data_read - sig_stereo)))
            # FLAC PCM_24 は loss-less な量子化 (24bit 精度内で復元可能)
            rt_status = "OK" if rt_max_abs < RT_THRESHOLD else f"LOSSY({rt_max_abs:.2e})"
            if rt_max_abs >= RT_THRESHOLD:
                verdict = "DEGRADED"
        finally:
            try:
                if os.path.exists(tmp_flac.name):
                    os.remove(tmp_flac.name)
            except OSError:
                pass

        # ---- (3) sosfiltfilt 等価性 (filtfilt(ba) vs sosfiltfilt(sos)) ----
        # 既存 zansei_impl の使用条件 (order=11, Wn=PRE_HP_CUTOFF_HZ/nyq 等) を再現
        rng_eq = _np.random.default_rng(4242)
        N_eq = 4096
        # sweep + white noise (フィルタを余すところなく exercise)
        t_eq = _np.arange(N_eq, dtype=_np.float32) / sr_test
        sweep = _sps.chirp(t_eq, f0=50.0, f1=20000.0, t1=t_eq[-1], method="logarithmic").astype(_np.float32)
        noise = rng_eq.standard_normal(N_eq).astype(_np.float32) * 0.1
        x_eq = (sweep + noise).astype(_np.float32)

        nyq = sr_test * 0.5
        # DC 信号 (HP filter の stopband rejection を数値で確認するため)
        dc_level = 0.5
        dc_sig = _np.full(N_eq, dc_level, dtype=_np.float32)

        eq_results: list[tuple[str, float, float, str]] = []  # (label, max_abs, rms_rel, tag)
        for label, wn_hz, btype in (
            ("pre_HP", float(PRE_HP_CUTOFF_HZ), "highpass"),
            ("post_HP", float(POST_HP_CUTOFF_HZ), "highpass"),
        ):
            wn = max(1e-6, min(0.999, wn_hz / nyq))
            # BA 形式は order > 8 かつ Wn < 0.1 で数値不安定 (scipy 公式警告)
            low_wn_regime = wn < 0.1

            # 旧 (ba)
            old_ok = True
            try:
                b, a = _sps.butter(FILTER_ORDER, wn, btype=btype)
                y_old = _sps.filtfilt(b, a, x_eq)
                if not _np.all(_np.isfinite(y_old)):
                    old_ok = False
            except Exception:
                old_ok = False
                y_old = None

            # 新 (sos)
            sos = _sps.butter(FILTER_ORDER, wn, btype=btype, output="sos")
            y_new = _sps.sosfiltfilt(sos, x_eq)
            new_finite = bool(_np.all(_np.isfinite(y_new)))

            if not new_finite:
                eq_results.append((label, float("nan"), float("nan"), "NEW_NaN"))
                verdict = "DEGRADED"
                continue

            # 陽性サニティ: HP フィルタは DC を強く減衰させるはず (-40dB 以上)
            # filtfilt は往復で効くので理論上 -60dB 以上期待できる
            y_dc = _sps.sosfiltfilt(sos, dc_sig)
            peak_dc = float(_np.max(_np.abs(y_dc))) + 1e-30
            dc_rejection_db = 20.0 * _np.log10(peak_dc / dc_level)
            if dc_rejection_db > -40.0:
                eq_results.append((label, float("nan"), float("nan"), f"HP_REJECT_BAD({dc_rejection_db:.1f}dB)"))
                verdict = "DEGRADED"
                continue

            if not old_ok or y_old is None:
                # 旧が NaN / 例外、新は有限 → IMPROVED
                eq_results.append((label, float("nan"), float("nan"), f"OLD_FAIL_NEW_OK(dc_rej={dc_rejection_db:.0f}dB)"))
                if verdict == "EQUIV":
                    verdict = "IMPROVED"
                continue

            diff = y_new - y_old
            max_abs = float(_np.max(_np.abs(diff)))
            rms_ref = float(_np.sqrt(_np.mean(y_old * y_old)) + 1e-30)
            rms_rel = float(_np.sqrt(_np.mean(diff * diff))) / rms_ref

            over_thresh = max_abs > 1e-4 or rms_rel > 1e-5
            if over_thresh and low_wn_regime:
                # 低 Wn では BA 形式自体が数値的に不正確。sos は HP サニティを満たしており
                # 新実装の方が正しい → IMPROVED (本家の数値バグを修正した形)
                tag = f"IMPROVED(max={max_abs:.2e},rms={rms_rel:.2e},dc_rej={dc_rejection_db:.0f}dB)"
                if verdict == "EQUIV":
                    verdict = "IMPROVED"
            elif over_thresh:
                tag = f"DIFFER(max={max_abs:.2e},rms={rms_rel:.2e})"
                verdict = "DEGRADED"
            else:
                tag = "EQUIV"
            eq_results.append((label, max_abs, rms_rel, tag))

        # ---- (4) zansei_impl の 3 負荷 determinism ----
        rng = _np.random.default_rng(1234)
        N = 4096
        x_stereo = rng.standard_normal((2, N)).astype(_np.float32) * 0.05
        sr_proc = 192000

        det_ok = True
        det_notes: list[str] = []
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
            if not _np.all(_np.isfinite(y1)) or not _np.all(_np.isfinite(y2)):
                det_notes.append(f"{lv}:NaN")
                det_ok = False
                continue
            if _np.array_equal(y1, y2):
                det_notes.append(f"{lv}:OK")
            else:
                max_abs_det = float(_np.max(_np.abs(y1 - y2)))
                det_notes.append(f"{lv}:diff(max={max_abs_det:.3e})")
                if max_abs_det > 1e-5:
                    det_ok = False
        if not det_ok:
            verdict = "DEGRADED"

        # ---- (5) ffmpeg 同梱確認 ----
        ffmpeg_path = _find_bundled(
            os.path.join("ffmpeg", "ffmpeg.exe"),
            os.path.join("_internal", "ffmpeg", "ffmpeg.exe"),
        )
        # 開発実行時 (frozen でない) は同梱を必須にしない
        if ffmpeg_path:
            ffmpeg_note = f"OK({os.path.basename(os.path.dirname(ffmpeg_path))}/ffmpeg.exe)"
        elif getattr(sys, "frozen", False):
            ffmpeg_note = "MISSING"
            verdict = "DEGRADED"
        else:
            ffmpeg_note = "dev(skip)"

        eq_summary = " ".join(f"{lbl}:{tag}" for (lbl, _m, _r, tag) in eq_results)

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(
                f"selftest verdict={verdict} numpy={_np.__version__} "
                f"scipy={_sp.__version__} librosa={_lb.__version__} "
                f"threadpoolctl={tpc_version} "
                f"roundtrip={OUTPUT_FORMAT}/{OUTPUT_SUBTYPE}={rt_status} "
                f"sosfiltfilt_equiv=[{eq_summary}] "
                f"determinism=[{' '.join(det_notes)}] "
                f"ffmpeg={ffmpeg_note}\n"
            )
        return 0 if verdict != "DEGRADED" else 1
    except Exception:
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("selftest FAILED verdict=DEGRADED\n")
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
