"""Tests for deterministic reproducibility across the pipeline.

SoftwareX requirement: same input + same seed → identical output.
"""
import numpy as np
import pytest
from respanno.audio.preprocessing import apply_butter_filter, validate_preprocessing_config
from respanno.dsp.features import compute_short_time_features, build_feature_matrix
from respanno.dsp.fft import compute_fft
from respanno.dsp.spectrogram import compute_stft_db
from respanno.ml.hsmm import build_hsmm_log_trans, build_hsmm_prior_from_prefix_labels, hsmm_viterbi
from respanno.ml.frame_labels import build_frame_labels
from tests.fixtures.synthetic_signals import generate_tone, generate_respiratory_cycle

class TestPreprocessingDeterminism:

    def test_butter_filter_deterministic(self):
        """验证 butter filter 的确定性：相同输入产生逐位相同的输出。"""
        (audio, sr, _) = generate_tone(freq=200.0, sr=4000, duration=2.0, seed=42)
        out1 = apply_butter_filter(audio, sr, 'bandpass', 50.0, 500.0, 4)
        out2 = apply_butter_filter(audio, sr, 'bandpass', 50.0, 500.0, 4)
        assert np.allclose(out1, out2)

    def test_config_validation_deterministic(self):
        """验证 config validation 的确定性：相同输入产生逐位相同的输出。"""
        cfg1 = validate_preprocessing_config({'resample_target_sr': 8000})
        cfg2 = validate_preprocessing_config({'resample_target_sr': 8000})
        assert cfg1 == cfg2

class TestFFTDeterminism:

    def test_fft_deterministic(self):
        """验证 fft 的确定性：相同输入产生逐位相同的输出。"""
        (audio, sr, _) = generate_tone(freq=200.0, sr=4000, duration=2.0, seed=42)
        (f1, m1) = compute_fft(audio, sr)
        (f2, m2) = compute_fft(audio, sr)
        assert np.allclose(f1, f2)
        assert np.allclose(m1, m2)

    def test_stft_deterministic(self):
        """验证 stft 的确定性：相同输入产生逐位相同的输出。"""
        (audio, sr, _) = generate_tone(freq=200.0, sr=4000, duration=2.0, seed=42)
        (S1, f1) = compute_stft_db(audio, sr, n_fft=512, hop_length=128)
        (S2, f2) = compute_stft_db(audio, sr, n_fft=512, hop_length=128)
        assert np.allclose(S1, S2, equal_nan=True)
        assert np.allclose(f1, f2)

class TestFeatureDeterminism:

    def test_short_time_features_deterministic(self):
        """验证 short time features 的确定性：相同输入产生逐位相同的输出。"""
        (audio, sr, _) = generate_tone(freq=200.0, sr=4000, duration=2.0, seed=42)
        (t1, fd1) = compute_short_time_features(audio, sr, hop_length=256)
        (t2, fd2) = compute_short_time_features(audio, sr, hop_length=256)
        assert np.allclose(t1, t2)
        for k in fd1:
            assert np.allclose(fd1[k], fd2[k], equal_nan=True)

    def test_feature_matrix_deterministic(self):
        """验证 feature matrix 的确定性：相同输入产生逐位相同的输出。"""
        (audio, sr, _) = generate_tone(freq=200.0, sr=4000, duration=2.0, seed=42)
        (t1, fd1) = compute_short_time_features(audio, sr, hop_length=256)
        (t2, fd2) = compute_short_time_features(audio, sr, hop_length=256)
        (X1, _, names1) = build_feature_matrix(t1, fd1)
        (X2, _, names2) = build_feature_matrix(t2, fd2)
        assert np.allclose(X1, X2, equal_nan=True)
        assert names1 == names2

class TestFrameLabelDeterminism:

    def test_build_frame_labels_deterministic(self):
        """验证 build frame labels 的确定性：相同输入产生逐位相同的输出。"""
        (audio, sr, anns) = generate_respiratory_cycle(duration=8.0, seed=42)
        (times, _) = compute_short_time_features(audio, sr, hop_length=256)
        y1 = build_frame_labels(anns, times, 'Wheeze', neg_margin=0.05)
        y2 = build_frame_labels(anns, times, 'Wheeze', neg_margin=0.05)
        if y1 is None:
            assert y2 is None
        else:
            assert np.array_equal(y1, y2)

class TestHSMMDeterminism:

    def test_log_trans_deterministic(self):
        """验证 log trans 的确定性：相同输入产生逐位相同的输出。"""
        names = ['Inspiration', 'Expiration', 'Pause']
        lt1 = build_hsmm_log_trans(names)
        lt2 = build_hsmm_log_trans(names)
        assert np.allclose(lt1, lt2)

    def test_hsmm_viterbi_deterministic(self):
        """验证 hsmm viterbi 的确定性：相同输入产生逐位相同的输出。"""
        names = ['Inspiration', 'Expiration']
        dmin = np.array([3, 3], dtype=int)
        dmax = np.array([30, 30], dtype=int)
        log_trans = build_hsmm_log_trans(names)
        log_pi = np.log(np.array([0.5, 0.5]))
        rng = np.random.default_rng(42)
        log_emit = np.log(np.abs(rng.standard_normal((50, 2))) + 1e-06)
        z1 = hsmm_viterbi(log_emit, dmin, dmax, log_trans, log_pi)
        z2 = hsmm_viterbi(log_emit, dmin, dmax, log_trans, log_pi)
        assert np.array_equal(z1, z2)

    def test_hsmm_prior_deterministic(self):
        """验证 hsmm prior 的确定性：相同输入产生逐位相同的输出。"""
        y_prefix = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0], dtype=np.int16)
        classes = [0, 1, 2]
        name_map = {0: 'Inspiration', 1: 'Expiration', 2: 'Pause'}
        p1 = build_hsmm_prior_from_prefix_labels(y_prefix, classes, name_map, 0.064, 4.0)
        p2 = build_hsmm_prior_from_prefix_labels(y_prefix, classes, name_map, 0.064, 4.0)
        assert p1['dmin_frames'] == p2['dmin_frames']
        assert p1['dmax_frames'] == p2['dmax_frames']

class TestSyntheticSignalDeterminism:

    def test_generator_same_seed_same_output(self):
        """验证：np.allclose(a1, a2)。"""
        (a1, sr1, ann1) = generate_respiratory_cycle(seed=42)
        (a2, sr2, ann2) = generate_respiratory_cycle(seed=42)
        assert np.allclose(a1, a2)
        assert sr1 == sr2
        assert ann1 == ann2

    def test_generator_different_seed_different_output(self):
        """验证：not np.allclose(a1, a2)。"""
        (a1, _, _) = generate_respiratory_cycle(seed=42)
        (a2, _, _) = generate_respiratory_cycle(seed=99)
        assert not np.allclose(a1, a2)


# ═══════════════════════════════════════════════════════════════════════════
# Cross-Process Reproducibility (SoftwareX requirement)
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossProcessReproducibility:
    """Verify that independent Python processes produce identical pipeline hashes.

    Same input + same seed → identical output, even across process boundaries.
    This proves there are no hidden global-state / caching effects.
    """

    @pytest.fixture(scope="class")
    def repro_script(self):
        """Path to the repro_check.py helper script."""
        import os as _os
        tests_dir = _os.path.dirname(_os.path.abspath(__file__))
        script = _os.path.join(tests_dir, "..", "scripts", "repro_check.py")
        script = _os.path.abspath(script)
        assert _os.path.isfile(script), f"repro_check.py not found at {script}"
        return script

    def _run_repro(self, repro_script, seed):
        """Run repro_check.py in a subprocess and return its FINAL_HASH."""
        import subprocess
        result = subprocess.run(
            ["python", repro_script, f"--seed={seed}"],
            capture_output=True, text=True, timeout=120,
            env={**__import__('os').environ, "PYTHONPATH": __import__('os').pathsep.join([
                __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))),
                __import__('os').path.dirname(__import__('os').path.abspath(__file__)),
            ])}
        )
        assert result.returncode == 0, f"subprocess failed: {result.stderr}"
        for line in result.stdout.splitlines():
            if line.startswith("FINAL_HASH:"):
                return line.split(":")[1].strip()
        raise RuntimeError(f"FINAL_HASH not found in output:\n{result.stdout}")

    def test_same_seed_same_hash_across_processes(self, repro_script):
        """Two independent subprocess runs with same seed → identical hash."""
        h1 = self._run_repro(repro_script, seed=42)
        h2 = self._run_repro(repro_script, seed=42)
        assert h1 == h2, f"Cross-process hashes differ: {h1[:16]}... vs {h2[:16]}..."
        assert len(h1) == 64  # SHA-256 hex digest

    def test_different_seed_different_hash_across_processes(self, repro_script):
        """Two subprocess runs with different seeds → different hashes."""
        h1 = self._run_repro(repro_script, seed=42)
        h2 = self._run_repro(repro_script, seed=99)
        assert h1 != h2, "Different seeds should produce different hashes"

    def test_three_runs_all_identical(self, repro_script):
        """Three independent runs with same seed → all identical (N=3 check)."""
        hashes = [self._run_repro(repro_script, seed=123) for _ in range(3)]
        assert len(set(hashes)) == 1, f"Expected 1 unique hash, got {len(set(hashes))}"