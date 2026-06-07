"""Cross-process reproducibility verification script.

Invoked by test_reproducibility.py via subprocess to verify that the
pipeline produces identical output across independent Python processes.
Usage:
    python scripts/repro_check.py --seed 42
Prints a SHA-256 hash of the concatenated pipeline outputs.
"""
import sys
import hashlib
import numpy as np

# Ensure project root is on path
import os as _os
_sys = sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from respanno.dsp.features import compute_short_time_features, build_feature_matrix
from tests.fixtures.synthetic_signals import generate_wheeze_episode, generate_tone


def hash_array(arr):
    """Return SHA-256 hex digest of a numpy array's bytes."""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def hash_dict(feat_dict):
    """Hash a feature dict by concatenating hashes of each array."""
    h = hashlib.sha256()
    for k in sorted(feat_dict.keys()):
        h.update(hash_array(np.asarray(feat_dict[k], dtype=float)).encode())
    return h.hexdigest()


def main(seed=42):
    h = hashlib.sha256()

    # 1. Generate synthetic wheeze episode
    audio, sr, anns = generate_wheeze_episode(duration=5.0, wheeze_start=1.0, wheeze_dur=2.5, seed=seed)
    h.update(hash_array(audio).encode())
    h.update(str(sr).encode())
    h.update(str(anns).encode())
    print(f"  audio hash: {hash_array(audio)[:16]}...")

    # 2. Extract features
    times, feat_dict = compute_short_time_features(audio, int(sr))
    h.update(hash_array(times).encode())
    h.update(hash_dict(feat_dict).encode())
    print(f"  features hash: {hash_dict(feat_dict)[:16]}...")

    # 3. Build feature matrix
    X_full, times2, full_names = build_feature_matrix(times, feat_dict)
    h.update(hash_array(X_full).encode())
    h.update(str(full_names).encode())
    print(f"  matrix hash: {hash_array(X_full)[:16]}...")

    # 4. Generate a tone → test FFT / STFT
    audio2, sr2, _ = generate_tone(freq=500.0, sr=4000, duration=2.0, seed=seed)
    from respanno.dsp.spectrogram import compute_stft_db
    from respanno.dsp.fft import compute_fft

    S_db, freqs = compute_stft_db(audio2, sr2, n_fft=256, hop_length=64)
    h.update(hash_array(S_db).encode())
    freqs2, mag = compute_fft(audio2, sr2)
    h.update(hash_array(freqs2).encode())
    h.update(hash_array(mag).encode())
    print(f"  STFT hash: {hash_array(S_db)[:16]}...")

    # 5. Test preprocessing
    from respanno.audio.preprocessing import apply_butter_filter
    filtered = apply_butter_filter(audio2, sr2, 'bandpass', 20, 1800, 4)
    h.update(hash_array(filtered).encode())
    print(f"  filter hash: {hash_array(filtered)[:16]}...")

    final_hash = h.hexdigest()
    print(f"\nFINAL_HASH: {final_hash}")
    return final_hash


if __name__ == "__main__":
    seed = 42
    for arg in sys.argv[1:]:
        if arg.startswith("--seed="):
            seed = int(arg.split("=")[1])
    main(seed)
