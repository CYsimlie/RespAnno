# Changelog

## [1.7.0] — 2026-05-13 (模块抽离完成)

### Added
- `respanno/labels/annotation_io.py` — 标签 IO 纯函数 (10 functions)
- `respanno/audio/preprocessing.py` — 音频预处理 (7 functions)
- `respanno/dsp/spectrogram.py` — STFT/频谱着色 (7 functions)
- `respanno/ml/hsmm.py` — HSMM Viterbi 解码 (6 functions)
- `respanno/dsp/features.py` — 56 短时特征 (6 functions)
- `tests/test_features_basic.py` — 15 特征测试
- `docs/software_architecture.md` — 架构文档
- `docs/testing.md` — 测试文档

### Changed
- 所有 TODO-skipped 测试已消除 (121 passed, 0 skipped)
- `README.md` 更新至当前工程化状态

### Notes
- 提取的模块均零 PyQt 依赖，纯 numpy/scipy/librosa
- 所有算法行为与 legacy 逐字节一致
- GUI 集成阶段尚未开始

---

## [1.6.6] — 2026-05-11 (工程化起点)

### Added
- 项目骨架：`legacy/`, `respanno/`, `tests/`, `docs/`, `demo_data/`, `examples/`, `screenshots/`
- `README.md`, `requirements.txt`, `environment.yml`, `pyproject.toml`, `LICENSE`, `CHANGELOG.md`
- `respanno/main.py` 启动入口
- 测试初稿：43 passed, 4 skipped

### Changed
- 原始 `1.6.6.py` 冻结至 `legacy/1.6.6.py`，不再编辑

---

## [1.6.6] — legacy

原始单文件实现，包含完整的 GUI、DSP、标注、ML 功能。
