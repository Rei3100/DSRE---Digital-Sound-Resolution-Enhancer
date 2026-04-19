# DSRE プロジェクト階層ルール

## 主対象
- **メインファイル**: `DSRE.py` (単一ファイル、約 300 行、PySide6 製 GUI + DSP)
- **設定ファイル**: なし (入出力は `INPUT_DIR` / `OUTPUT_DIR` 定数でハードコード: `C:\Audio\DSRE` / `C:\Audio\DSRE\Output`)
- **Python**: 3.11 のみ対応 (PySide6 は 3.11 に install 済み、`py -3.11` で検証)

## 由来・フォーク事情
- 本家: [x1aoqv/DSRE---Digital-Sound-Resolution-Enhancer](https://github.com/x1aoqv/DSRE---Digital-Sound-Resolution-Enhancer) (507 行)
- このフォーク: [Rei3100/DSRE---Digital-Sound-Resolution-Enhancer](https://github.com/Rei3100/DSRE---Digital-Sound-Resolution-Enhancer) (ChatGPT で凝縮済み、現在約 300 行)
- 参考 (別派生 v2.0): [Urabewe/DSRE v2.0 Enhanced](https://github.com/Urabewe/DSRE---Digital-Sound-Resolution-Enhancer-English) (2000 行超、UI 改善・リトライ等のみ参考)
- **音質は本家の zansei_impl が好み** → 音響処理ロジック (freq_shift_*, zansei_impl, safe_butter) は**触らない**
- v2.0 の psychoacoustic_enhancer / multiband_exciter は**取り込まない** (音が変わるため)

## 絶対ルール
1. **UI ノータッチ**: `MainWindow` クラスのウィジェット構成・レイアウト・ボタン名 (「開始」「一時停止」「取消」)・setWindowTitle・resize(320, 180) は**1 文字も変えない**
2. **音響処理ロジック温存**: `zansei_impl`, `freq_shift_mono/multi`, `safe_butter` の計算式は改変禁止 (リファクタで名前変更しただけの等価変換のみ OK)
3. **本家影響ゼロ**: `git remote` は `origin` (Rei3100/DSRE) のみ。**`upstream` remote を絶対に追加しない**。本家への PR/push は永久禁止
4. **`run_hidden` は `CREATE_NO_WINDOW` 単独で使う** (STARTUPINFO は不要、yt-dlp GUI v17.2 と揃える)
5. **Windows subprocess**: 必ず `creationflags=CREATE_NO_WINDOW`、コマンドプロンプトをポップさせない
6. **INPUT_DIR / OUTPUT_DIR はハードコード維持** (個人ワークフローで固定、UI から変更させない方針)
7. **新機能は別セッションで plan → 承認 → 実装**。今のスコープはリファクタ + バグ修正のみ

## 定数中央集約 (ファイル冒頭)
- `INPUT_DIR` / `OUTPUT_DIR`
- `HARMONIC_LAYERS` / `HARMONIC_DECAY` / `PRE_HP_CUTOFF_HZ` / `POST_HP_CUTOFF_HZ` / `TARGET_SR` / `FILTER_ORDER`
- `@dataclass(frozen=True) class DSREParams` + `PARAMS = DSREParams()` インスタンス

音響パラメータを変更する場合は、上記定数 + DSREParams の両方を更新する。

## 依存管理
- **ランタイム**: `requirements.txt` (PySide6/numpy/scipy/librosa/resampy/soundfile/send2trash のみ、UTF-8)
- **ビルド**: `requirements-dev.txt` (pyinstaller 系のみ)
- librosa が引きずる audioread/numba/llvmlite/pooch/soxr/joblib 等は明示列挙しない (pip に任せる)

## Claude Code 運用
- 編集は Claude Code 側で完結、ユーザーは GitHub 運用の知識ゼロ前提
- push は**ユーザーが「push して」と明示したとき**のみ
- コミットはオプション単位で分割 (`refactor:` / `fix:` / `deps:` / `tooling:` プレフィックス)
- 編集後の自動検証 (py_compile) は `~/.claude/settings.json` の PostToolUse hook が走る (yt_dlp_gui.py と DSRE.py の両方が対象)

## 関連サブエージェント
- `dsre-specialist` (DSP 勘所・本家差分・音質改変禁止ルール熟知)
- `code-reviewer` / `bug-hunter` / `refactor-finder` は汎用として使える
