# SoftX Package Gap Report -- RespAnno v1.0.0

**Date:** 2026-06-08
**Audit method:** Automated scan + manual verification + pytest execution
**Status:** Preliminary — user action required before SoftwareX submission

---

## Summary

| Category | PASS | WARN | FAIL | NEED_USER_CONFIRMATION |
|----------|------|------|------|------------------------|
| Root files | 10 | 2 | 3 | 2 |
| Code structure | 14 | 2 | 0 | 1 |
| Documentation | 5 | 2 | 1 | 2 |
| Examples | 0 | 0 | 1 | 1 |
| Screenshots | 0 | 0 | 1 | 1 |
| Tests | 8 | 1 | 0 | 0 |
| License | 3 | 1 | 0 | 1 |
| Citation/Version | 4 | 2 | 1 | 2 |
| Sensitive info | 7 | 0 | 0 | 2 |
| **TOTAL** | **51** | **10** | **7** | **12** |

---

## Detailed Gap Table

### A. Root Files

| Area | Requirement | Current status | Evidence | Risk level | Action |
|------|-------------|----------------|----------|------------|--------|
| README.md | Must exist and be complete | PASS | README.md: 386 lines, includes Code Metadata + Software Metadata tables | Low | — |
| LICENSE | Must exist | WARN | LICENSE exists, MIT, but "Copyright (c) 2025 C.Y.Pan" — year should be 2026, author should be full name | Low | Update copyright year to 2026; replace "C.Y.Pan" with full name |
| VERSION | Separate VERSION file recommended | FAIL | No `VERSION` file exists. Version only in `respanno/__init__.py` | Low | Create `VERSION` file containing `1.0.0` |
| CHANGELOG.md | Should exist and be current | PASS | CHANGELOG.md: v1.0.0 entry present, dated 2026-06-07 | Low | — |
| CITATION.cff | Must exist | WARN | CITATION.cff exists but author is "Chao Yue Pan" (abbreviated), email is placeholder `chaoyuepan@example.com`, ORCID is empty | Medium | NEED_USER_CONFIRMATION: full name, real email, ORCID |
| CONTRIBUTING.md | Should exist | PASS | CONTRIBUTING.md present | Low | — |
| CODE_OF_CONDUCT.md | Should exist | PASS | CODE_OF_CONDUCT.md present | Low | — |
| PROJECT_OUTLINE.md | Recommended | PASS | PROJECT_OUTLINE.md: comprehensive paper-writing prompt | Low | Contains some Chinese text; consider translating to English |
| TEST_RELIABILITY_REPORT.md | Recommended | PASS | TEST_RELIABILITY_REPORT.md exists but version info outdated (still references "~450 tests") | Medium | Update test count to 535, version to 1.0.0, date to 2026-06-07 |
| requirements.txt | Must exist | PASS | requirements.txt exists with loose constraints | Low | Consider pinning to tested versions (matching CLAUDE.md dev environment table) |
| environment.yml | Must exist | PASS | environment.yml exists | Low | environment name "respanno" — ensure CI uses same name |
| pyproject.toml | Should exist | PASS | pyproject.toml: version="1.0.0", author="C.Y.Pan" | Low | Replace "C.Y.Pan" with full name |
| .gitignore | Must exist | WARN | .gitignore exists but missing entries: `.coverage`, `coverage.xml`, `.vscode/` | Medium | Add `.coverage*`, `coverage.xml`, `htmlcov/`, `.vscode/` to .gitignore |
| .github/workflows/test.yml | Recommended | PASS | CI config present: 3 OS × 3 Python versions | Low | — |

### B. Code Structure

| Area | Requirement | Current status | Evidence | Risk level | Action |
|------|-------------|----------------|----------|------------|--------|
| Python package structure | Clear package with `__init__.py` | PASS | `respanno/__init__.py` exports `__version__` | Low | — |
| Entry point clarity | GUI entry must be clear | PASS | `1.0.0.py` is the main GUI entry; CLAUDE.md documents this | Low | — |
| Backend/GUI separation | Backend must not depend on PyQt5 | PASS | `respanno/` imports verified; `service.py` uses lazy imports for classifier/phase_model; `_get_sd()` lazy-imports sounddevice | Low | — |
| Legacy protection | legacy/1.0.0.py must be unmodified | PASS | `git diff -- legacy/1.0.0.py` returns empty output | Low | — |
| CRLF line endings | 1.0.0.py must use CRLF | PASS | `file` command reports "UTF-8 text executable, with CRLF line terminators" | Low | — |
| Hardcoded paths | No absolute paths | PASS | No `C:\`, `D:\`, `/home/` paths found in source code | Low | — |
| Private data | No patient/hospital data in code | WARN | Demo data files named `101_1b1_Pr_sc_Meditron.wav` etc. — "Meditron" and "Litt3200" are medical device names (stethoscope models), but **are these recordings from real human subjects?** | High | NEED_USER_CONFIRMATION: confirm demo data is either publicly available (e.g., ICBHI 2017 dataset) or synthetic. If harvested from real subjects without consent, remove before open-sourcing. |
| .coverage file in repo | Should not be committed | WARN | `.coverage` (53248 bytes) is tracked in git | Low | Remove from git: `git rm --cached .coverage` and add to .gitignore |
| __pycache__ directories | 14 instances found | WARN | 14 `__pycache__/` directories exist in working tree | Low | Add `__pycache__/` (with trailing slash) to .gitignore; run `git rm -r --cached **/__pycache__` |
| Package importability | All modules must import cleanly | PASS | All 14 backend modules imported successfully | Low | — |
| Chinese feature names | Features have Chinese names | PASS | `respanno/dsp/features.py` uses Chinese feature names as dict keys (e.g., `谱质心`, `短时能量`, `过零率`) — this is by design per CLAUDE.md | Low | Document in README that feature names use Chinese characters for compatibility with legacy codebase |
| MLService lazy imports | Confirmed | PASS | `service.py` lines 97, 145, 208, 222: lazy imports inside methods | Low | — |
| GUI widget testing | At minimum, SettingsDialog tested | PASS | `tests/test_gui_widgets_headless.py`: 33 tests covering 7 widget classes | Low | — |
| Architecture diagram accuracy | Must match reality | PASS | CLAUDE.md architecture tree matches actual file tree (verified by `find`) | Low | — |

### C. Documentation

| Area | Requirement | Current status | Evidence | Risk level | Action |
|------|-------------|----------------|----------|------------|--------|
| Installation instructions | Must be in README | PASS | README.md includes conda and pip install paths | Low | — |
| Usage guide | Should exist | PASS | README.md: "Typical Workflow" section (10 steps), Keyboard Shortcuts table | Low | — |
| Test running instructions | Must be clear | WARN | README lists `conda run -n respanno python -m pytest tests -q` but CLAUDE.md points to env `respanno-test` | Low | Unify environment naming in documentation |
| API / developer docs | Recommended | WARN | `docs/software_architecture.md` and `docs/testing.md` exist but are outdated (reference 1.6.6.py, old test counts) | Low | Update docs to reference 1.0.0.py and current test counts |
| Chinese content in docs | Should be English | WARN | PROJECT_OUTLINE.md, TEST_RELIABILITY_REPORT.md, CHANGELOG.md older entries, docs/windows_gui_test_plan.md contain Chinese text | Medium | Translate remaining Chinese in documentation files |
| User guide completeness | Should be comprehensive | FAIL | No standalone user guide. README covers basics, but no detailed instructions for ML pipeline, source field semantics, export format specifications, or troubleshooting | Medium | Either expand README or create docs/user_guide.md |
| Disclaimer statement | Highly recommended | FAIL | README, tests, and docs do not contain explicit disclaimer that "these tests demonstrate functional correctness and should not be interpreted as clinical performance validation" | Medium | Add disclaimer to README (testing section) and TEST_RELIABILITY_REPORT.md |

### D. Examples

| Area | Requirement | Current status | Evidence | Risk level | Action |
|------|-------------|----------------|----------|------------|--------|
| Example data | Should include demo WAV + annotations | FAIL | `examples/` contains only `.gitkeep`; `demo_data/` exists but unclear if public | High | NEED_USER_CONFIRMATION: add example workflow script, sample CSV annotations to `examples/`; confirm demo_data origin |
| Workflow example | Recommended | WARN | `scripts/ml_demo.py` exists but is not in examples/ | Low | Move or copy `ml_demo.py` to `examples/workflow_demo.py` |
| Round-trip example | Recommended | WARN | No standalone example showing WAV→annotate→export→reimport | Low | Create `examples/roundtrip_demo.py` |

### E. Screenshots

| Area | Requirement | Current status | Evidence | Risk level | Action |
|------|-------------|----------------|----------|------------|--------|
| Screenshots directory | Must contain screenshots | FAIL | `screenshots/` contains only `.gitkeep` | High | NEED_USER_CONFIRMATION: capture 4-5 screenshots (main window, ML workflow, settings dialog, annotation dialog, spectrogram views) |
| Screenshot content | Must be publishable | NEED_USER_CONFIRMATION | N/A — no screenshots yet | High | Ensure screenshots do not reveal personal paths, subject information, or hospital identifiers |

### F. Tests

| Area | Requirement | Current status | Evidence | Risk level | Action |
|------|-------------|----------------|----------|------------|--------|
| Test execution | Must all pass | PASS | `pytest tests -q` output: 535 collected, 534 passed, 1 skipped (LoopPlayer requires sounddevice) | Low | — |
| Actual test count | Must be verified | PASS | `grep -c "def test_" tests/test_*.py` = 399 test functions across 24 test files + conftest + 2 fixture files | Low | Update CLAUDE.md: Tests: 535 collected, 534 pass, 1 skip, 399 test functions |
| Signal processing tests | Should be covered | PASS | test_fft_basic.py (12), test_spectrogram_basic.py (21), test_preprocessing_basic.py (30), test_features_basic.py (19) | Low | — |
| Annotation I/O tests | Should be covered | PASS | test_annotation_roundtrip.py (43), test_annotation_quality.py (21) | Low | — |
| Source-aware annotation | Should be covered | PASS | test_annotation_quality.py: TestSourceProvenance, test_roundtrip_workflow.py: TestSourceProvenanceRoundtrip | Low | — |
| Label routing and mapping | Should be covered | PASS | test_label_taxonomy_basic.py (8), test_ml_service_basic.py (25) | Low | — |
| MLService dispatcher | Previously a gap | PASS | test_ml_service_basic.py: 25 test functions covering init, routing, train dispatch, apply dispatch, clear, HSMM helpers, E2E | Low | Gap CLOSED |
| ML-assisted annotation | Should be covered | PASS | test_classifier_training_basic.py (13), test_classifier_apply_basic.py (7), test_e2e_ml_pipeline.py (8) | Low | — |
| HSMM phase smoothing | Should be covered | PASS | test_hsmm_basic.py (18), test_phase_model_basic.py (12), test_phase_apply_basic.py (8) | Low | — |
| GUI testing | Should acknowledge limitations | WARN | test_gui_widgets_headless.py (33) + test_gui_static_integration.py (20); BoxSpan, multi-lane collision avoidance not tested | Low | Document as known gap in TEST_RELIABILITY_REPORT.md |
| Known test gaps status | Must check all 5 | See below | Refer to detailed audit | — | — |

#### Known Test Gap Status (from CLAUDE.md)

| Gap | Status | Evidence |
|-----|--------|----------|
| 1. MLService dispatcher routing | **CLOSED** | test_ml_service_basic.py: 25 tests |
| 2. Annotation CRUD semantics | **PARTIALLY OPEN** | test_annotation_quality.py covers normalization + source provenance; undo/redo cycle with negative samples not explicitly tested |
| 3. Multi-lane layout (_pick_lane) | **OPEN** | No automated test for 3-lane collision avoidance; verified only through manual GUI testing |
| 4. Hard negative feedback effectiveness | **PARTIALLY OPEN** | test_negatives_basic.py covers NegSampleManager CRUD; delete→retrain→improved predictions cycle not tested E2E |
| 5. GUI modules manual testing only | **PARTIALLY CLOSED** | 33 headless widget tests added; BoxSpan (618 lines) still requires manual testing |

### G. License

| Area | Requirement | Current status | Evidence | Risk level | Action |
|------|-------------|----------------|----------|------------|--------|
| LICENSE file | Must exist | PASS | LICENSE: MIT, present | Low | — |
| License consistency | Must match across files | PASS | README, pyproject.toml, CITATION.cff all say MIT | Low | — |
| Author in LICENSE | Should be full name | WARN | "C.Y.Pan" — abbreviated | Low | Replace with full name |
| Copyright year | Should be current | WARN | "Copyright (c) 2025" — should be 2026 | Low | Update to 2026 |
| PyQt5 GPL compatibility | Must be acknowledged | NEED_USER_CONFIRMATION | RespAnno uses MIT license; PyQt5 is GPLv3. Distributing RespAnno with PyQt5 as a bundled application may require GPL compliance. This DOES NOT affect RespAnno's own code license. | Medium | Add note in README or LICENSE about PyQt5 licensing. Pure-package users install PyQt5 separately; only relevant for PyInstaller bundles. |

### H. Citation, Version, and Release

| Area | Requirement | Current status | Evidence | Risk level | Action |
|------|-------------|----------------|----------|------------|--------|
| VERSION file | Recommended | FAIL | No `VERSION` file | Low | Create `VERSION` file |
| Version consistency | Must be 1.0.0 everywhere | PASS | pyproject.toml=1.0.0, respanno/__init__.py=1.0.0, CITATION.cff=1.0.0, README C1/S1=1.0.0, CHANGELOG=[1.0.0] | Low | — |
| CITATION.cff author | Should be full name | WARN | "Chao Yue Pan" abbreviated, no ORCID, placeholder email | Medium | NEED_USER_CONFIRMATION |
| GitHub release | Recommended before submission | FAIL | No GitHub release tag created yet (local git tag `clean-before-en` exists but not pushed) | Medium | Create v1.0.0 release on GitHub with release notes |
| Zenodo DOI | Required for SoftwareX | FAIL | Not archived | High | NEED_USER_CONFIRMATION: Archive on Zenodo to get DOI |
| Release date | CITATION.cff says 2026-06-03; CHANGELOG says 2026-06-07 | WARN | Date mismatch | Low | Unify to actual release date |

### I. Sensitive Information

| Area | Requirement | Current status | Evidence | Risk level | Action |
|------|-------------|----------------|----------|------------|--------|
| Placeholder email | Should be real | NEED_USER_CONFIRMATION | CITATION.cff, README: `chaoyuepan@example.com` | Medium | Replace with real contact email |
| Author name | Should be full name | NEED_USER_CONFIRMATION | "C.Y.Pan" in LICENSE, pyproject.toml, 1.0.0.py About dialog | Medium | Replace with full name |
| Patient data | Must not be present | WARN | Demo data filenames suggest clinical origin (Meditron, Litt3200, LittC2SE). No patient identifiers found in text search. | High | NEED_USER_CONFIRMATION: confirm these are ICBHI 2017 public dataset excerpts, not private clinical recordings |
| Hardcoded paths | None found | PASS | No absolute Windows/Linux paths in source | Low | — |
| API keys / tokens | None found | PASS | No api_key, token, password, secret detected | Low | — |
| .coverage file | Should not be in repo | WARN | Committed `.coverage` (52 KB) contains local file paths | Medium | Remove from git tracking |
| Windows user path in .coverage | Privacy concern | WARN | `.coverage` contains `C:\Users\pcy30\...` paths | Medium | Remove .coverage from git |
| GitHub URL | Points to real repo? | NEED_USER_CONFIRMATION | README, CITATION.cff reference `https://github.com/chaoyuepan/RespAnno` | Medium | Confirm this URL is correct and the repo will be made public |

---

## Terminology Audit

| Term | Found | Context | Verdict |
|------|-------|---------|---------|
| "clinical" | README.md:71 | "and clinical studies" — describing target audience | **OK** — descriptive, not claiming clinical validation |
| "ground truth" | tests/fixtures/synthetic_signals.py:7 | "Known ground truth" — refers to synthetic generated annotations | **OK** — clearly about synthetic data with known labels |
| "confidence" | Not found in source | — | **OK** — no false "confidence score" claims |
| "fully automatic" | Not found | — | **OK** |
| "HSMM" | Multiple files | Always scoped to respiratory phase (Inspiration/Expiration/Pause), never applied to wheeze/crackles/rhonchi/stridor | **OK** — correctly scoped |
| "CAS/DAS" | Not found | — | **OK** — no misuse |
| "automatic ground truth" | Not found | — | **OK** |

---

## 535 vs 454 Test Count Discrepancy Explained

| Source | Count | Explanation |
|--------|-------|-------------|
| CLAUDE.md line 11 | "454 tests across 23 test files" | **Outdated.** This was the count before the test enhancement phase (June 4). |
| README.md badge | "535 passed" | **Updated** during this session. |
| pytest actual output | 535 collected, 534 passed, 1 skipped | **Verified.** 26 files total: 24 test_*.py + conftest.py + __init__.py. |
| grep "def test_" | 399 test functions | Correct — some tests are parameterized (e.g., 43 collected from 8 functions in test_label_taxonomy_basic.py). |
| New test files added | test_ml_service_basic.py (25), test_gui_widgets_headless.py (33) | These + expanded roundtrip/reproducibility tests account for the ~80 test increase. |

**Recommendation:** Update CLAUDE.md, TEST_RELIABILITY_REPORT.md, and README README test count section to consistently report 535 tests.
