# Changelog

## [1.0.0] — 2026-06-07

### First public release

- 535 tests across 26 test modules (534 pass, 1 skip), 85% code coverage
- MLService dispatcher with full test coverage (24 tests, 96% line coverage)
- Headless GUI widget tests covering 7 widget classes (37 tests)
- Cross-process reproducibility verification via SHA-256 pipeline hashing
- Performance baseline tests (report-only mode, non-blocking in CI)
- CI/CD: GitHub Actions on 3 OS × 3 Python versions with coverage reports
- All comments and docstrings translated to English
- CHANGELOG, CITATION.cff, CONTRIBUTING.md, CODE_OF_CONDUCT.md, PROJECT_OUTLINE.md

## [1.6.6] — 2026-06-03

### Added
- `respanno/ml/service.py` — ML pipeline dispatcher (train/apply routing)
- Lazy sounddevice import for headless CI compatibility
- CITATION.cff — academic citation metadata
- CONTRIBUTING.md — contribution guidelines
- CODE_OF_CONDUCT.md — community standards

### Changed
- Default STFT parameters: n_fft 512→256, hop_length 256→64
- All Chinese comments rewritten to professional style
- Cleaned 13 dead sklearn/lightgbm imports from main file
- Main file (1.0.0.py) now contains only AudioViewer class (~2443 lines)
- Updated architecture tree and test counts in README

### Fixed
- Duplicate `import os, sys` cleaned up
- Version number consistency across README / pyproject.toml / __init__.py

---

## [1.6.6-beta] — 2026-05-28

### Added
- `respanno/ml/classifier.py` — Binary LightGBM event classifier (380 lines)
- `respanno/ml/phase_model.py` — Phase model training & application (430 lines)
- `respanno/ml/label_taxonomy.py` — Label-to-pipeline routing (101 lines)
- `respanno/ml/frame_labels.py` — Frame-level training label builder
- 427 tests across 22 test modules (all pass)

### Changed
- MLService shrunk from ~995 lines to 251 lines (74% reduction)
- 1.0.0.py shrunk from 3711 to ~2700 lines
- 8 import bugs fixed in extracted modules

---

## [1.6.6-alpha] — 2026-05-13 (模块抽离完成)

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
- 原始 `1.0.0.py` 冻结至 `legacy/1.0.0.py`，不再编辑

---

## [1.6.6] — legacy

原始单文件实现，包含完整的 GUI、DSP、标注、ML 功能。
