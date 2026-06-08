# Real-Data ML-Assisted Annotation Evaluation Results

**RespAnno v1.0.0 | 2026-06-08**

## Purpose

Demonstrate that RespAnno's ML-assisted annotation pipeline can learn from
a small number of manually-reviewed segments and accurately locate additional
instances of the same event type in the remainder of a real respiratory
recording.

## Method

- **Signal**: `demo_data/4000Hz/101_1b1_Al_sc_Meditron.wav` — 20 s respiratory
  recording at 4000 Hz
- **Ground truth**: `demo_data/events/101_1b1_Al_sc_Meditron_events.csv` —
  22 phase annotations (11 Inspiration + 11 Expiration, alternating)
- **Pipeline**: GT loaded via `respanno.labels.annotation_io`, audio via
  `respanno.audio.preprocessing`, features via `respanno.dsp.features`,
  training/inference via `respanno.ml.phase_model` (HSMM Viterbi).
- **Split**: first N=2 segments per label used for training ("reviewed"),
  remaining segments withheld for evaluation
- **Metric**: Intersection-over-Union (IoU) > 0.3 counted as a match

## Results

| Label      | Reviewed | Withheld | ML candidates | IoU matches | Recall |
|------------|----------|----------|---------------|-------------|--------|
| Inspiration | 2 | 9 | 9 | 9 | 9/9 (100%) |
| Expiration | 2 | 9 | 9 | 9 | 9/9 (100%) |
| **Total** | **4** | **18** | **18** | **18** | **18/18 (100%)** |

### Per-segment detail

**Inspiration** (IoU range: 0.74–1.00)

| Withheld GT | ML candidate | IoU | HIT/MISS |
|-------------|--------------|-----|----------|
| [4.06–4.51] | [4.10–4.54] | 0.87 | HIT |
| [5.68–6.27] | [5.68–6.29] | 0.98 | HIT |
| [7.62–8.17] | [7.62–8.08] | 0.84 | HIT |
| [9.46–9.94] | [9.44–9.92] | 0.93 | HIT |
| [11.23–11.71] | [11.23–11.68] | 0.93 | HIT |
| [13.09–13.54] | [13.09–13.54] | 1.00 | HIT |
| [14.64–15.24] | [14.64–15.09] | 0.74 | HIT |
| [16.72–17.20] | [16.72–17.25] | 0.91 | HIT |
| [18.67–19.12] | [18.67–19.12] | 1.00 | HIT |

**Expiration** (IoU range: 0.86–0.97)

| Withheld GT | ML candidate | IoU | HIT/MISS |
|-------------|--------------|-----|----------|
| [4.51–5.68] | [4.54–5.73] | 0.93 | HIT |
| [6.27–7.62] | [6.29–7.58] | 0.97 | HIT |
| [8.17–9.46] | [8.19–9.38] | 0.92 | HIT |
| [9.94–11.23] | [9.86–11.23] | 0.94 | HIT |
| [11.71–13.09] | [11.73–13.06] | 0.97 | HIT |
| [13.54–14.64] | [13.55–14.62] | 0.97 | HIT |
| [15.24–16.72] | [15.14–16.69] | 0.91 | HIT |
| [17.20–18.67] | [17.26–18.56] | 0.88 | HIT |
| [19.12–19.95] | [19.04–20.00] | 0.86 | HIT |

## Discussion

- **No false positives.** The HSMM Viterbi decoder's Insp/Exp alternation
  constraint eliminates the fragmentation artifacts that a plain binary
  classifier would produce on rhythmic signals.
- **No false negatives.** All 18 withheld segments were recovered.
- **Training data was minimal** — only 4 manually-annotated segments
  (2 Inspiration + 2 Expiration, combined duration ~2.5 s) served as the
  entire training set. The remaining 18 segments (combined duration ~11 s)
  were successfully auto-labeled.
- **Boundary precision is high** (mean IoU = 0.92 for Inspiration,
  0.93 for Expiration), reflecting the HSMM duration prior learned from
  the reviewed prefix.

## Notes

- The HSMM phase model is used **only** for respiratory phase labels
  (Inspiration/Expiration/Pause). Adventitious sound labels (Wheeze,
  Crackles, Rhonchi, Stridor) use the LightGBM binary classifier pipeline.
- The evaluation script (`examples/real_data_eval.py`) is reproducible
  with any WAV + CSV pair in RespAnno's standard annotation format.
