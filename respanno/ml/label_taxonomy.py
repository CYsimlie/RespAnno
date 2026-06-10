"""Label taxonomy for routing annotations to ML pipelines.

Constants and helper functions for classifying labels into one of three
pipeline kinds: phase (Inspiration/Expiration), abnormal_sound (default),
or other_event (speech/cough/etc).
"""

# ── Label taxonomy ──────────────────────────────────────────────────────────

PHASE_LABELS = {
    "inspiration", "expiration", "pause",
    "insp", "exp", "inhale", "exhale", "exspiration",
    "吸气", "呼气", "Pause", "停顿"
}

OTHER_EVENT_LABELS = {
    "speech", "talk", "talking", "speaking", "voice", "vocal", "whisper",
    "cough", "coughing", "sneeze", "snore", "laugh", "cry", "swallow",
    "throat", "说话", "讲话", "咳嗽", "咳",
    "noise", "artifact", "movement", "rub", "stethoscope", "background", "normal"
}

ABNORMAL_SOUND_KIND = "abnormal_sound"
OTHER_EVENT_KIND = "other_event"
PHASE_KIND = "phase"


# ── Public API ──────────────────────────────────────────────────────────────

def label_kind(label):
    """Route a label string to one of three pipeline kinds.

    Returns one of: PHASE_KIND, OTHER_EVENT_KIND, or ABNORMAL_SOUND_KIND (default).
    """
    lab = str(label).strip().lower()
    if lab in PHASE_LABELS:
        return PHASE_KIND
    if lab in OTHER_EVENT_LABELS:
        return OTHER_EVENT_KIND
    return ABNORMAL_SOUND_KIND


def clear_ml_annotations(viewer, label):
    """Delete all machine annotations for `label` from `viewer`.

    Removes BoxSpan objects from viewer._spans (via viewer.delete_annotation)
    and nullifies entries in viewer.annotations whose source is 'ml' and
    whose label matches.
    """
    if hasattr(viewer, "ml"):
        return

    spans_to_delete = []

    for span in list(getattr(viewer, "_spans", [])):
        idx = getattr(viewer, "_span2idx", {}).get(span, None)
        if idx is None:
            continue

        if idx < 0 or idx >= len(viewer.annotations):
            continue

        item = viewer.annotations[idx]
        if item is None:
            continue

        try:
            if len(item) == 3:
                s, e, t = item
                src = "manual"
            elif len(item) >= 4:
                s, e, t, src = item[:4]
            else:
                continue
        except Exception:
            continue

        if src == "ml" and str(t).strip().lower() == str(label).strip().lower():
            spans_to_delete.append(span)

    for sp in spans_to_delete:
        try:
            viewer.delete_annotation(sp, record_negative=False, push_undo=False)
        except Exception:
            pass

    for i, item in enumerate(viewer.annotations):
        if item is None:
            continue
        try:
            if len(item) == 3:
                continue
            elif len(item) >= 4:
                s, e, t, src = item[:4]
            else:
                continue
        except Exception:
            continue

        if src == "ml" and str(t).strip().lower() == str(label).strip().lower():
            viewer.annotations[i] = None
