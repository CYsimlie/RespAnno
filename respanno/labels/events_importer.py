"""Automatic import of matching _events annotation files for WAV audio.

EventsFileIndexer scans directories for annotation files matching the
pattern `<wav_base>_events.<ext>`, caches the index, and imports them
into the AudioViewer on demand.
"""

import os


DEFAULT_AUTO_IMPORT_CFG = {
    "file_format": "auto",
    "file_suffix": "_events",
    "delimiter": "auto",
    "custom_delimiter": "",
    "skip_header_lines": 0,
    "start_col": 1,
    "end_col": 2,
    "label_col": 3,
    "source_col": 4,
    "json_start_key": "start",
    "json_end_key": "end",
    "json_label_key": "label",
    "json_source_key": "source",
}


class EventsFileIndexer:
    """Scans folders and caches `<wav_base>_events.<ext>` mappings.

    Parameters
    ----------
    viewer : AudioViewer
        The main viewer instance, used to read UI settings and call
        ``finalize_annotation`` when importing.
    """

    def __init__(self, viewer):
        self._viewer = viewer
        self._index = {}   # {folder(abs): {wav_base_lower: events_path}}
        self._parse_cache = {}  # {events_path(abs): (mtime, rows)}

    # ── settings helpers ──────────────────────────────────────────────────

    def _get_settings(self):
        """Merge auto-import settings from viewer with defaults."""
        cfg = dict(DEFAULT_AUTO_IMPORT_CFG)
        try:
            viewer_cfg = dict(
                getattr(self._viewer, "auto_label_import_settings", {}) or {}
            )
        except Exception:
            viewer_cfg = {}
        cfg.update(viewer_cfg)
        return cfg

    def _candidate_extensions(self):
        """Return list of file extensions to search, by priority."""
        cfg = self._get_settings()
        fmt = str(cfg.get("file_format", "auto")).strip().lower()
        if fmt in {"csv", "txt", "json"}:
            return [fmt]
        return ["csv", "txt", "json"]

    # ── index management ──────────────────────────────────────────────────

    def build_index(self, folder):
        """Scan `folder` for matching events files; cache the mapping."""
        try:
            folder = os.path.abspath(folder)
        except Exception:
            return
        if folder in self._index:
            return

        cfg = self._get_settings()
        suffix = str(cfg.get("file_suffix", "_events") or "_events").lower()
        exts = self._candidate_extensions()
        ext_rank = {ext: i for i, ext in enumerate(exts)}

        mapping = {}
        try:
            for ent in os.scandir(folder):
                if not ent.is_file():
                    continue
                name = ent.name
                low = name.lower()
                for ext in exts:
                    tail = f"{suffix}.{ext}"
                    if low.endswith(tail):
                        base = name[:-len(tail)].lower()
                        old_path = mapping.get(base)
                        if old_path is None:
                            mapping[base] = ent.path
                        else:
                            try:
                                old_ext = (
                                    os.path.splitext(old_path)[1]
                                    .lower()
                                    .lstrip(".")
                                )
                                if ext_rank.get(ext, 999) < ext_rank.get(
                                    old_ext, 999
                                ):
                                    mapping[base] = ent.path
                            except Exception:
                                pass
                        break
        except Exception:
            mapping = {}

        self._index[folder] = mapping

    # ── path resolution ───────────────────────────────────────────────────

    def resolve_path(self, wav_path):
        """Return the events file path matching `wav_path`, or None."""
        if not wav_path or not isinstance(wav_path, str):
            return None
        wav_path = os.path.abspath(wav_path)
        folder = os.path.dirname(wav_path)
        wav_base = os.path.splitext(os.path.basename(wav_path))[0]
        key = wav_base.lower()

        self.build_index(folder)
        mapping = self._index.get(folder, {})

        p = mapping.get(key)
        if p and os.path.isfile(p):
            return p

        # fallback: existence check for late-created files
        cfg = self._get_settings()
        suffix = str(cfg.get("file_suffix", "_events") or "_events")
        for ext in self._candidate_extensions():
            cand = os.path.join(folder, f"{wav_base}{suffix}.{ext}")
            if os.path.isfile(cand):
                mapping[key] = cand
                self._index[folder] = mapping
                return cand

        return None

    # ── file parsing ──────────────────────────────────────────────────────

    def parse_file(self, events_path):
        """Parse an events file, returning a list of annotation dicts."""
        from respanno.labels.annotation_io import read_annotations

        cfg = self._get_settings()
        return read_annotations(events_path, cfg)

    def parse_file_cached(self, events_path):
        """Parse with mtime-based caching."""
        if isinstance(events_path, str):
            try:
                p = os.path.abspath(events_path)
                mtime = os.path.getmtime(p)
            except Exception:
                return []
        else:
            return []

        cached = self._parse_cache.get(p)
        if (
            cached
            and isinstance(cached, tuple)
            and len(cached) == 2
            and cached[0] == mtime
        ):
            return cached[1]

        rows = self.parse_file(p)
        self._parse_cache[p] = (mtime, rows)
        return rows

    # ── import ────────────────────────────────────────────────────────────

    def auto_import(self, wav_path):
        """Find and import matching events file without showing dialogs."""
        events_path = self.resolve_path(wav_path)
        if not events_path:
            return

        rows = self.parse_file_cached(events_path)
        if not rows:
            return

        n_ok = 0
        for ann in rows:
            try:
                s, e = float(ann["start"]), float(ann["end"])
                lab = str(ann["label"])
                src = str(ann.get("source", "manual") or "manual")
                if e <= s:
                    continue
                self._viewer.finalize_annotation(s, e, lab, source=src)
                n_ok += 1
            except Exception:
                continue

        try:
            self._viewer.statusBar().showMessage(
                f"Auto-imported {n_ok} annotations from "
                f"{os.path.basename(events_path)}",
                2000,
            )
        except Exception:
            pass
