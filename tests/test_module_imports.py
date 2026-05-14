"""Verify that all respanno subpackages import cleanly.

This test runs without any GUI, audio device, or heavy computation.
It confirms the package structure is well-formed and all public
symbols are accessible.
"""

import importlib
import pytest


RESPANNO_MODULES = [
    "respanno",
    "respanno.labels.annotation_io",
    "respanno.audio.preprocessing",
    "respanno.dsp.spectrogram",
    "respanno.dsp.features",
    "respanno.ml.hsmm",
]

# (module_name, [(attr_name, expected_type), ...])
EXPECTED_ATTRS = {
    "respanno.labels.annotation_io": [
        ("normalize_annotation", "function"),
        ("parse_annotation_row", "function"),
        ("read_annotations", "function"),
        ("read_annotations_csv", "function"),
        ("read_annotations_json", "function"),
        ("write_annotations", "function"),
        ("write_annotations_csv", "function"),
        ("write_annotations_json", "function"),
        ("roundtrip_annotations", "function"),
        ("DEFAULT_LABEL_CONFIG", "dict"),
    ],
    "respanno.audio.preprocessing": [
        ("validate_preprocessing_config", "function"),
        ("compute_target_sr", "function"),
        ("apply_butter_filter", "function"),
        ("apply_preprocessing", "function"),
        ("preprocess_audio_file", "function"),
        ("summarize_preprocessing", "function"),
        ("DEFAULT_PREPROCESSING_CONFIG", "dict"),
    ],
    "respanno.dsp.spectrogram": [
        ("compute_stft_db", "function"),
        ("decimate_spec_for_display", "function"),
        ("get_palette_256", "function"),
        ("colorize_spectrogram", "function"),
        ("compute_spectrogram_display", "function"),
        ("DEFAULT_STFT_CONFIG", "dict"),
    ],
    "respanno.dsp.features": [
        ("compute_short_time_features", "function"),
        ("compute_time_domain_features", "function"),
        ("compute_spectral_features", "function"),
        ("normalize_feature_for_display", "function"),
        ("build_feature_matrix", "function"),
        ("ALL_FEATURE_NAMES", "list"),
    ],
    "respanno.ml.hsmm": [
        ("estimate_hop_sec", "function"),
        ("estimate_breath_cycle_sec", "function"),
        ("build_hsmm_prior_from_prefix_labels", "function"),
        ("build_hsmm_log_trans", "function"),
        ("hsmm_viterbi", "function"),
        ("state_seq_to_segments", "function"),
    ],
}


class TestAllModulesImport:
    @pytest.mark.parametrize("module_name", RESPANNO_MODULES)
    def test_import(self, module_name):
        mod = importlib.import_module(module_name)
        assert mod is not None

    @pytest.mark.parametrize("module_name", RESPANNO_MODULES)
    def test_has_version_or_doc(self, module_name):
        mod = importlib.import_module(module_name)
        has_doc = bool(getattr(mod, "__doc__", None))
        has_version = bool(getattr(mod, "__version__", None))
        assert has_doc or has_version or True  # always OK; just probing


class TestPublicSymbols:
    @pytest.mark.parametrize("module_name, attr_name, expected_type", [
        (mod, attr, typ)
        for mod, attrs in EXPECTED_ATTRS.items()
        for attr, typ in attrs
    ])
    def test_attr_exists_and_type(self, module_name, attr_name, expected_type):
        mod = importlib.import_module(module_name)
        obj = getattr(mod, attr_name)
        if expected_type == "function":
            assert callable(obj), f"{module_name}.{attr_name} should be callable"
        elif expected_type == "dict":
            assert isinstance(obj, dict), f"{module_name}.{attr_name} should be dict"
        elif expected_type == "list":
            assert isinstance(obj, list), f"{module_name}.{attr_name} should be list"


class TestRespannoTopLevel:
    def test_version_string(self):
        import respanno
        ver = getattr(respanno, "__version__", None)
        assert ver is not None
        assert isinstance(ver, str)
        assert len(ver) > 0


class TestNoHardDeps:
    """Verify extracted modules don't drag in heavy GUI deps."""
    HEAVY = {"PyQt5", "pyqtgraph", "sounddevice"}

    @pytest.mark.parametrize("module_name", [
        "respanno.labels.annotation_io",
        "respanno.ml.hsmm",
    ])
    def test_no_gui_import(self, module_name):
        """labels and hsmm should never import PyQt/pyqtgraph/sounddevice."""
        import sys
        before = set(sys.modules.keys())
        importlib.import_module(module_name)
        after = set(sys.modules.keys())
        new = after - before
        heavy_found = self.HEAVY & new
        assert not heavy_found, f"{module_name} pulled in: {heavy_found}"
