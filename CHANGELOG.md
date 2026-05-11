# Changelog

## [1.6.6] — 2026-05-11 (工程化起点)

### Added
- 项目骨架：`legacy/`, `respanno/`, `tests/`, `docs/`, `demo_data/`, `examples/`, `screenshots/`
- `README.md`, `requirements.txt`, `environment.yml`, `pyproject.toml`, `LICENSE`, `CHANGELOG.md`
- `respanno/main.py` 启动入口
- 测试初稿：`tests/conftest.py`, `test_preprocessing_basic.py`, `test_annotation_roundtrip.py`, `test_spectrogram_basic.py`, `test_hsmm_basic.py`

### Changed
- 原始 `1.6.6.py` 冻结至 `legacy/1.6.6.py`，不再编辑

### Note
- 当前阶段仅建立工程骨架与测试护栏，所有功能仍由 legacy 程序驱动
- 后续阶段将逐步从 legacy 抽取模块至 respanno 各子包

---

## [1.6.6] — legacy

原始单文件实现，包含完整的 GUI、DSP、标注、ML 功能。
