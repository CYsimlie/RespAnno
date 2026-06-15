# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['1.0.0.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        # respanno backend modules (lazy-imported by MLService, not detected statically)
        'respanno.ml.classifier',
        'respanno.ml.phase_model',
        'respanno.ml.hsmm',
        'respanno.ml.label_taxonomy',
        'respanno.ml.frame_labels',
        'respanno.ml.negatives',
        'respanno.dsp.features',
        'respanno.dsp.spectrogram',
        'respanno.dsp.fft',
        'respanno.audio.preprocessing',
        'respanno.labels.annotation_io',
        'respanno.labels.events_importer',
        # sklearn subpackages that PyInstaller's hook may miss
        'sklearn.feature_selection._mutual_info',
        'sklearn.utils._openmp_helpers',
        'sklearn.ensemble._hist_gradient_boosting',
        # sounddevice PortAudio backend
        'sounddevice',
        '_sounddevice_data',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='RespAnno',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
