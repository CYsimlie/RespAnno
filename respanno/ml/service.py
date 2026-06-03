"""MLService: centralized dispatcher for ML training/inference pipelines.

Routes train/apply calls to the appropriate backend module based on label taxonomy
(phase vs abnormal breath sounds vs other events).
"""


class MLService:
    """Centralized ML training/inference dispatcher."""

    # --- Label taxonomy routing (phase vs abnormal breath sounds vs other events) ---
    # 1) Breath phases (HSMM post-processing, used only for Inspiration/Expiration/Pause)
    PHASE_LABELS = {
        "inspiration", "expiration", "pause",
        "insp", "exp", "inhale", "exhale", "exspiration",
        "吸气", "呼气", "Pause", "停顿"
    }
    # 2) Other events (speech, cough, etc.) — currently share the same binary
    #    classifier as abnormal sounds, but routed via model_kind for future
    #    replacement with dedicated models.
    OTHER_EVENT_LABELS = {
        "speech", "talk", "talking", "speaking", "voice", "vocal", "whisper",
        "cough", "coughing", "sneeze", "snore", "laugh", "cry", "swallow", "throat",
        "说话", "讲话", "咳嗽", "咳",
        "noise", "artifact", "movement", "rub", "stethoscope", "background",
        "normal"
    }
    # 3) Abnormal breath sounds (wheeze, crackle, rhonchi, stridor, etc.) — default branch
    ABNORMAL_SOUND_KIND = "abnormal_sound"
    OTHER_EVENT_KIND = "other_event"
    PHASE_KIND = "phase"

    def __init__(self, owner):
        self.owner = owner

    def clear_ml_annotations_for_label(self, label):
        """Delete all machine annotations for `label`, delegating to label_taxonomy."""
        from respanno.ml.label_taxonomy import clear_ml_annotations

        clear_ml_annotations(self.owner, label)

    # --- Per-label pipeline routing ---
    def _label_kind(self, label):
        """Route label to pipeline kind, delegating to label_taxonomy."""
        from respanno.ml.label_taxonomy import label_kind

        return label_kind(label)

    def train_model_for_label(self, label, min_pos_frames=30, neg_pos_ratio=5,
                              random_state=None):
        """Dispatcher entrypoint used by UI."""
        kind = self._label_kind(label)
        if kind == MLService.PHASE_KIND:
            return self.train_phase_model_for_label(
                label, min_pos_frames=min_pos_frames,
                neg_pos_ratio=neg_pos_ratio, random_state=random_state,
            )
        if kind == MLService.OTHER_EVENT_KIND:
            return self.train_other_event_model_for_label(
                label, min_pos_frames=min_pos_frames,
                neg_pos_ratio=neg_pos_ratio, random_state=random_state,
            )
        return self.train_abnormal_sound_model_for_label(
            label, min_pos_frames=min_pos_frames,
            neg_pos_ratio=neg_pos_ratio, random_state=random_state,
        )

    def train_abnormal_sound_model_for_label(self, label, min_pos_frames=30,
                                             neg_pos_ratio=5, random_state=None):
        """Train an abnormal-sound label model (binary classifier).

        Currently shares the same binary training pipeline as other_event,
        differentiated via model_kind for future replacement.
        """
        return self.train_event_model_for_label(
            label, min_pos_frames=min_pos_frames,
            neg_pos_ratio=neg_pos_ratio, random_state=random_state,
            model_kind=MLService.ABNORMAL_SOUND_KIND,
        )

    def train_other_event_model_for_label(self, label, min_pos_frames=30,
                                          neg_pos_ratio=5, random_state=None):
        """Train an other-event label model (speech/cough/etc.).

        Currently shares the same binary training pipeline as abnormal_sound,
        differentiated via model_kind for future replacement.
        """
        return self.train_event_model_for_label(
            label, min_pos_frames=min_pos_frames,
            neg_pos_ratio=neg_pos_ratio, random_state=random_state,
            model_kind=MLService.OTHER_EVENT_KIND,
        )

    def train_phase_model_for_label(self, label, min_pos_frames=30,
                                    neg_pos_ratio=5, random_state=None):
        """Train phase model, delegating to respanno.ml.phase_model."""
        from respanno.ml.phase_model import train_phase_model

        return train_phase_model(
            self.owner, label,
            min_pos_frames=min_pos_frames,
            neg_pos_ratio=neg_pos_ratio,
            random_state=random_state,
        )

    def apply_model_for_label_on_unreviewed(self, label, min_dur_sec=0.05):
        """Dispatcher entrypoint used by UI."""
        kind = self._label_kind(label)
        if kind == MLService.PHASE_KIND:
            return self.apply_phase_model_for_label_on_unreviewed(
                label, min_dur_sec=min_dur_sec)
        if kind == MLService.OTHER_EVENT_KIND:
            return self.apply_other_event_model_for_label_on_unreviewed(
                label, min_dur_sec=min_dur_sec)
        return self.apply_abnormal_sound_model_for_label_on_unreviewed(
            label, min_dur_sec=min_dur_sec)

    def apply_abnormal_sound_model_for_label_on_unreviewed(self, label,
                                                           min_dur_sec=0.05):
        """Auto-annotate unreviewed regions for abnormal-sound labels.

        Currently shares the same binary post-processing as other_event,
        differentiated via model_kind for future replacement.
        """
        return self.apply_event_model_for_label_on_unreviewed(
            label, min_dur_sec=min_dur_sec,
            expected_model_kinds={MLService.ABNORMAL_SOUND_KIND, "event"},
        )

    def apply_other_event_model_for_label_on_unreviewed(self, label,
                                                        min_dur_sec=0.05):
        """Auto-annotate unreviewed regions for other-event labels.

        Currently shares the same binary post-processing as abnormal_sound,
        differentiated via model_kind for future replacement.
        """
        return self.apply_event_model_for_label_on_unreviewed(
            label, min_dur_sec=min_dur_sec,
            expected_model_kinds={MLService.OTHER_EVENT_KIND, "event"},
        )

    def apply_phase_model_for_label_on_unreviewed(self, label,
                                                  min_dur_sec=0.05):
        """Apply phase model, delegating to respanno.ml.phase_model."""
        from respanno.ml.phase_model import apply_phase_model

        return apply_phase_model(self.owner, label, min_dur_sec=min_dur_sec)

    # ------------------- HSMM helpers (phase only) -------------------
    def _estimate_hop_sec(self, times, viewer=None):
        """Estimate frame hop in seconds, delegating to hsmm module."""
        from respanno.ml.hsmm import estimate_hop_sec

        sr = hop_length = None
        try:
            if viewer is not None:
                sv = getattr(viewer, "sr", None)
                hv = getattr(viewer, "hop_length", None)
                if (sv is not None and hv is not None
                        and float(sv) > 0 and float(hv) > 0):
                    sr = float(sv)
                    hop_length = float(hv)
        except Exception:
            pass
        return estimate_hop_sec(times=times, sr=sr, hop_length=hop_length)

    def _estimate_breath_cycle_sec(self, seg_I, seg_E):
        """Estimate breath-cycle duration, delegating to hsmm module."""
        from respanno.ml.hsmm import estimate_breath_cycle_sec

        return estimate_breath_cycle_sec(seg_I, seg_E, default=3.0)

    def _build_hsmm_prior_from_prefix_labels(self, y_prefix, classes_,
                                             state_id_to_name, hop_sec,
                                             cycle_sec):
        """Build HSMM duration priors, delegating to hsmm module."""
        from respanno.ml.hsmm import build_hsmm_prior_from_prefix_labels

        return build_hsmm_prior_from_prefix_labels(
            y_prefix, classes_, state_id_to_name, hop_sec, cycle_sec,
        )

    def _build_hsmm_log_trans(self, state_names):
        """Build HSMM log-transition matrix, delegating to hsmm module."""
        from respanno.ml.hsmm import build_hsmm_log_trans

        return build_hsmm_log_trans(state_names)

    def _hsmm_viterbi(self, log_emit, dmin, dmax, log_trans, log_pi):
        """HSMM Viterbi decoder, delegating to hsmm module."""
        from respanno.ml.hsmm import hsmm_viterbi

        return hsmm_viterbi(log_emit, dmin, dmax, log_trans, log_pi)

    def _state_seq_to_segments(self, times, idx_unr, z_state_ids,
                               target_state_id, min_dur_sec):
        """Convert state sequence to segments, delegating to hsmm module."""
        from respanno.ml.hsmm import state_seq_to_segments

        return state_seq_to_segments(
            times, idx_unr, z_state_ids, target_state_id, min_dur_sec,
        )

    def train_event_model_for_label(self, label, min_pos_frames=20,
                                    neg_pos_ratio=5, random_state=None,
                                    model_kind="event"):
        """Train event model, delegating to respanno.ml.classifier."""
        from respanno.ml.classifier import train_event_model

        return train_event_model(
            self.owner, label,
            min_pos_frames=min_pos_frames,
            neg_pos_ratio=neg_pos_ratio,
            random_state=random_state,
            model_kind=model_kind,
        )

    def apply_event_model_for_label_on_unreviewed(self, label,
                                                  min_dur_sec=0.05,
                                                  expected_model_kinds=None):
        """Apply event model, delegating to respanno.ml.classifier."""
        from respanno.ml.classifier import apply_event_model

        return apply_event_model(
            self.owner, label,
            min_dur_sec=min_dur_sec,
            expected_model_kinds=expected_model_kinds,
        )
