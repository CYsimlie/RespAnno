# SoftX User Required Edits — RespAnno v1.0.0

**Date:** 2026-06-08
**Instructions:** Every item marked NEED_USER_CONFIRMATION or ACTION REQUIRED must be resolved by the repository owner (C.Y.Pan / Chao Yue Pan) before SoftwareX submission. Items are ordered by urgency.

---

## CRITICAL — Must resolve before submission

| # | Item | Why needed | Current evidence | User action required |
|---|------|-----------|------------------|----------------------|
| 1 | **Author full name** | SoftwareX Code Metadata table (C9), LICENSE, CITATION.cff, pyproject.toml all require author identification | `LICENSE`: "C.Y.Pan"; `pyproject.toml`: `{name = "C.Y.Pan"}`; `CITATION.cff`: "Chao Yue Pan"; `1.0.0.py` L2178: "Author: C.Y.Pan" | Replace "C.Y.Pan" with full legal name in: LICENSE, pyproject.toml, CITATION.cff, 1.0.0.py About dialog. Decide on consistent formatting (e.g., "Chao-Yue Pan" or "Chao Yue Pan") |
| 2 | **Contact email** | SoftwareX requires support contact | `CITATION.cff`: `chaoyuepan@example.com`; `README.md`: same placeholder | Replace with a real academic/institutional email address |
| 3 | **ORCID** | Recommended by SoftwareX for author disambiguation | `CITATION.cff`: `orcid: ''` (empty) | Add ORCID identifier if you have one |
| 4 | **Demo data origin** | Must confirm demo data is publicly distributable | `demo_data/` contains files named like `101_1b1_Pr_sc_Meditron.wav` — "Meditron" is a 3M Littmann stethoscope model. Naming convention matches ICBHI 2017 challenge dataset. | Confirm these are excerpts from the ICBHI 2017 public dataset (permissible to redistribute). If these are private clinical recordings, remove immediately and replace with synthetic or public-domain data. |
| 5 | **Screenshots** | SoftwareX requires screenshots showing the tool in operation | `screenshots/` contains only `.gitkeep` | Capture 4-5 screenshots: (1) main window with loaded WAV and annotations, (2) ML training workflow, (3) settings dialog with preprocessing tab, (4) annotation label dialog, (5) spectrogram/FFT/features view. Save as PNG to `screenshots/`. Ensure no personal paths or subject info visible. |
| 6 | **GitHub repository URL** | SoftwareX Code Metadata C2 requires permanent repository link | README, CITATION.cff reference `https://github.com/chaoyuepan/RespAnno` | Confirm this URL is correct and the repo will be made public. If a different URL, update all files. |
| 7 | **Zenodo DOI** | SoftwareX Code Metadata C3 and Software Metadata S3 require DOI | Not yet archived | Create public GitHub release v1.0.0 → connect to Zenodo → obtain DOI → fill into README C3/S3 fields |

---

## HIGH — Strongly recommended before submission

| # | Item | Why needed | Current evidence | User action required |
|---|------|-----------|------------------|----------------------|
| 8 | **Create VERSION file** | Repository best practice; some tools expect it | No `VERSION` file exists | Create `VERSION` containing `1.0.0` |
| 9 | **Unify release date** | Consistency across metadata | `CITATION.cff`: date-released: 2026-06-03; `CHANGELOG.md`: v1.0.0 dated 2026-06-07 | Set to actual GitHub release date |
| 10 | **Update copyright year** | LICENSE accuracy | `LICENSE`: "Copyright (c) 2025" | Change to "Copyright (c) 2026" |
| 11 | **Clean .gitignore** | Prevent accidental commits of build artifacts | `.coverage` (52KB, contains local paths) tracked in git; 14 `__pycache__/` dirs present | Add `.coverage*`, `coverage.xml`, `htmlcov/`, `__pycache__/`, `.vscode/` to .gitignore. Run `git rm --cached .coverage`. |
| 12 | **Remove development scripts** | 13 translation/debug scripts in `scripts/` are one-time-use tools, not part of the software product | `scripts/_*.py` (13 files) | Delete or move to a separate `dev/` directory excluded from the repository. Keep: `functional_test.py`, `ml_demo.py`, `repro_check.py`. |
| 13 | **Update TEST_RELIABILITY_REPORT.md** | Currently shows outdated test count "~450" | TEST_RELIABILITY_REPORT.md line 3 | Update to 535 tests, 26 test files, 534 pass, 1 skip. Add disclaimer: "These tests demonstrate functional correctness and reproducibility of the software infrastructure. They should not be interpreted as clinical performance validation." |
| 14 | **Expand examples/** | SoftwareX expects executable examples | `examples/` is empty | Add: `workflow_demo.py` (10-line script loading WAV→extracting features), `ml_demo.py` (copy from scripts/), `sample_annotations.csv` (example annotation file) |

---

## MEDIUM — Improves review quality

| # | Item | Why needed | Current evidence | User action required |
|---|------|-----------|------------------|----------------------|
| 15 | **Add disclaimer to README** | Prevent misinterpretation of test results | No disclaimer found | Add to README Testing section: "These tests demonstrate functional correctness, file-format robustness, and reproducibility of the software infrastructure. They should not be interpreted as clinical performance validation or as a substitute for independent detection model evaluation." |
| 16 | **Translate remaining Chinese in documentation** | Reviewers may not read Chinese | `PROJECT_OUTLINE.md`, `TEST_RELIABILITY_REPORT.md`, `docs/windows_gui_test_plan.md`, `docs/windows_manual_test_record_template.md` contain Chinese text | Translate to English or provide bilingual versions |
| 17 | **Unify test runner environment name** | Avoid confusion | README: `respanno`; CLAUDE.md: `respanno`; but actual working env is `respanno-test` | Decide on one environment name (recommend: `respanno`). Update `environment.yml` and README accordingly. |
| 18 | **Consider pinning dependency versions** | Reproducibility | `requirements.txt` uses loose constraints (`numpy>=1.21`); actual tested versions are in CLAUDE.md dev environment table | Add `requirements-pinned.txt` with tested versions (numpy==1.24.4, scipy==1.10.1, etc.) for exact reproducibility |
| 19 | **Update outdated docs** | `software_architecture.md` and `testing.md` reference old 1.6.6.py paths | Multiple docs files | Update file references from 1.6.6.py to 1.0.0.py; update test counts |
| 20 | **PyQt5 license note** | MIT + GPL compatibility for bundled distributions | RespAnno = MIT; PyQt5 = GPLv3. This does NOT affect the source code license, but PyInstaller bundles may have implications. | Add brief note in README: "RespAnno is MIT-licensed. Note that PyQt5 is GPLv3. If you distribute a bundled executable (e.g., via PyInstaller), your distribution may need to comply with GPLv3 terms." |
| 21 | **Remove/verify demo_data/events/ contents** | `demo_data/events/` seems to contain WAV files, not annotation files | Directory listing shows WAV copies in events/ | If these are misplaced, reorganize. The `events/` directory should contain annotation files matching the OriginFs WAVs. |

---

## LOW — Nice to have

| # | Item | Why needed | Current evidence | User action required |
|---|------|-----------|------------------|----------------------|
| 22 | **Cross-platform testing confirmation** | CLAUDE.md pending item #6 | CI covers Ubuntu, Windows, macOS (3 OS × 3 Python versions) — but this only confirms automated tests pass. GUI manual testing status unknown for macOS/Linux. | If GUI has been tested on macOS/Linux, add notes. If not, clearly state "GUI manually tested on Windows 10/11 only." |
| 23 | **Supplementary video** | Optional but helpful for GUI tools | Not present | Consider recording a 2-3 minute screencast showing the annotation workflow |
| 24 | **Record Windows GUI functional test results** | `docs/windows_manual_test_record_template.md` is a blank template | Template is not filled in | Fill in the template with actual test results from a Windows GUI session |
| 25 | **Update CLAUDE.md test count** | AI assistant documentation should be accurate | CLAUDE.md still says "454 tests, 23 files" in Test Suite section | Update to match actual: 535 collected, 534 pass, 1 skip, 24 test files |

---

## Summary: Items requiring manual work (counted)

| Priority | Count | Items |
|----------|-------|-------|
| CRITICAL | 7 | #1-#7 |
| HIGH | 7 | #8-#14 |
| MEDIUM | 7 | #15-#21 |
| LOW | 4 | #22-#25 |
| **TOTAL** | **25** | |

---

## Quick-start: what to do first

1. **Right now:** Fill in author name, email, ORCID (#1, #2, #3) — this unlocks LICENSE/CITATION/README/1.0.0.py About dialog
2. **Right now:** Confirm demo data source (#4) — if uncertain, replace with synthetic
3. **Next:** Capture screenshots (#5) — this is the most time-consuming manual step
4. **Next:** Create GitHub release → Zenodo DOI (#6, #7)
5. **Then:** Run the cleanup items in order (#8-#14 should take ~1 hour total)
6. **Finally:** Translation and polish (#15-#25)
