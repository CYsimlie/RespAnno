# RespAnno User Guide — Writing Plan

**Version:** v1.0.0
**Target audience:** Researchers and clinicians working with respiratory sound recordings
**Estimated final length:** 15-20 pages (with screenshots)

---

## Chapter structure

### 1. Introduction (1 page)
- What RespAnno is
- Who it is for
- Key capabilities overview
- What RespAnno is NOT (not a diagnostic tool, not a clinical device)

### 2. Installation (2 pages)
- Prerequisites (Python 3.9+, conda recommended)
- Option A: conda environment (`conda env create -f environment.yml`)
- Option B: pip + venv (`pip install -r requirements.txt`)
- Launching the application (`python 1.0.0.py` or `python -m respanno.main`)
- Windows notes (CRLF, Qt plugin path)
- Linux notes (headless CI limitation, sounddevice/PortAudio)
- Troubleshooting common issues

### 3. Quick Start (2 pages)
- Load your first WAV file (Ctrl+O)
- Understand the interface: STFT spectrogram (top), waveform (middle), annotation track (bottom)
- Play/pause audio (Space)
- Navigate: Left/Right seek, Up/Down next file
- Your first annotation: drag in the annotation track → select label → commit
- Export annotations (Ctrl+E)

### 4. Interface Overview (3 pages)
- Menu bar: File, Settings, Help
- Toolbar buttons and their functions
- The three-panel layout (page stack / waveform / annotations)
- Switching views: STFT → FFT → Short-Time Features
- Color map switching (Heatmap / Grayscale)
- Status bar information
- ML toolbar (label dropdown, Train, Auto-label, Clear Negatives)

### 5. Audio Preprocessing (2 pages)
- Why resample? (default 4000 Hz)
- Butterworth filtering options (bandpass/lowpass/highpass/bandstop)
- Configuring preprocessing in Settings (Ctrl+P)
- Understanding the Preprocessing tab
- Zero-phase filtering (filtfilt)
- When preprocessing takes effect (on next WAV load)

### 6. Manual Annotation (3 pages)
- Built-in label presets (9 labels: Wheeze, Crackles, Pleural Rub, Rhonchi, Stridor, Speech, Cough, Inspiration, Expiration)
- Custom labels
- Creating annotations: drag-to-select workflow
- Editing annotations: double-click to enter edit mode, drag handles, Enter/Esc
- Deleting annotations: select + Delete
- Undo (Ctrl+Z)
- Right-click context menu (Play, Delete, Accept ML)
- The three-lane layout (collision avoidance)

### 7. Annotation Source Tracking (1 page)
- What is `source`? (provenance of each annotation)
- Source values: manual, ml, auto_accepted, auto_edited, merged
- How source affects visual appearance (solid vs dashed)
- How source affects training data selection
- Accepting ML annotations (Ctrl+A): changes source from `ml` → `auto_accepted`

### 8. Import & Export (2 pages)
- Exporting annotations: Ctrl+E (CSV/TXT/JSON)
- Importing annotations: Ctrl+I
- File format specifications
- Column mapping configuration
- Auto-import of matching `_events` files
- Configuring auto-import in Settings

### 9. Machine-Learning-Assisted Annotation (4 pages)
- Overview: how ML-assisted labeling works
- The training workflow:
  - 1. Manually annotate some regions (creates positive examples)
  - 2. Delete false ML predictions (creates hard negative samples)
  - 3. Select a label → Ctrl+T (Train Model)
  - 4. Ctrl+M (Auto-label Unreviewed)
  - 5. Review and accept/reject ML candidates
- Hard negative sample feedback loop
- What "Clear Negatives" does and when to use it
- **IMPORTANT**: ML-assisted annotations are **candidate suggestions**, not ground truth. The human annotator always makes the final decision.
- Per-label model storage

### 10. Respiratory Phase Modeling (1 page)
- How phase modeling differs from event detection
- HSMM (Hidden Semi-Markov Model) for Inspiration/Expiration/Pause
- HSMM is used ONLY for phase labels, NOT for wheeze/crackles/rhonchi/stridor
- Duration priors and physiological constraints

### 11. Short-Time Features View (1 page)
- The 56 short-time features
- Selecting features to display (up to 5)
- Feature color assignment
- Time-axis synchronization with STFT

### 12. Keyboard Shortcuts Reference (1 page)
- Complete shortcut table (from CLAUDE.md)

### 13. Configuration Reference (1 page)
- All Settings dialog tabs explained
- STFT tab
- Display tab
- Preprocessing tab
- Auto Label Import tab
- Short-Time Features tab

### Appendix A: Example Workflow (2 pages)
- Step-by-step annotated example with screenshots
- Load demo WAV → annotate → train → auto-label → review → export

### Appendix B: Troubleshooting (1 page)
- "PortAudio not found" (sounddevice issue on Linux)
- "No module named PyQt5" (install PyQt5)
- WAV file not loading (format compatibility)
- Chinese character display issues
- Memory issues with very long recordings

---

## Writing progress tracking

| Chapter | Status | Estimated hours | Screenshots needed |
|---------|--------|-----------------|-------------------|
| 1. Introduction | not started | 0.5 | 0 |
| 2. Installation | not started | 1 | 0 |
| 3. Quick Start | not started | 1 | 2 |
| 4. Interface Overview | not started | 2 | 1 |
| 5. Audio Preprocessing | not started | 1 | 1 |
| 6. Manual Annotation | not started | 2 | 2 |
| 7. Annotation Source Tracking | not started | 0.5 | 0 |
| 8. Import & Export | not started | 1 | 0 |
| 9. ML-Assisted Annotation | not started | 2 | 2 |
| 10. Respiratory Phase Modeling | not started | 0.5 | 0 |
| 11. Short-Time Features View | not started | 0.5 | 0 |
| 12. Keyboard Shortcuts | ✅ done | 0 | 0 |
| 13. Configuration Reference | not started | 1 | 1 |
| Appendix A: Example Workflow | not started | 1 | 2 |
| Appendix B: Troubleshooting | not started | 0.5 | 0 |
| **TOTAL** | | **~14.5 h** | **11 screenshots** |

---

## Source material available

- README.md: installation, quick start, shortcuts, architecture
- CLAUDE.md: keyboard shortcuts, architecture, ML pipeline details
- PROJECT_OUTLINE.md: comprehensive module and feature descriptions
- 1.0.0.py: actual GUI behavior (reference for accurate descriptions)
- screenshots/: (to be captured)
- demo_data/: sample respiratory WAV files for examples

## Key principles

1. **Every claim must be verifiable** — describe only what the software actually does
2. **Screenshots for every major workflow step** — users learn visually
3. **ML-assisted, not automatic** — consistently use "ML candidate" / "auto-label" language, never "automatic detection" or "ground truth generation"
4. **HSMM scope is limited** — explicitly state HSMM is for phase labels only
5. **Clinical disclaimer** — state clearly that RespAnno is a research annotation tool, not a diagnostic device
