"""Pure annotation I/O helpers (no PyQt / librosa / sounddevice dependency).

All functions operate on a standard dict format:

    {
        "start": float,
        "end": float,
        "label": str,
        "source": str  (default "manual")
    }

The ``config`` dict throughout this module mirrors the structure stored
in ``AudioViewer.auto_label_import_settings`` (see legacy 1.6.6.py).
"""

from __future__ import annotations

import csv
import io
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

DEFAULT_LABEL_CONFIG: Dict[str, Any] = {
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


def _build_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = dict(DEFAULT_LABEL_CONFIG)
    if overrides:
        cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# 1. normalize_annotation
# ---------------------------------------------------------------------------

def normalize_annotation(
    item: Union[tuple, list, dict, None],
    target_source: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Convert a legacy-style annotation item into the standard dict form.

    Accepted inputs:
    - 3-tuple / 3-list  → (start, end, label)   → source="manual"
    - 4-tuple / 4-list  → (start, end, label, source)
    - dict with "start"/"end"/"label" keys       → source default "manual"

    Returns None for invalid items.
    """
    if item is None:
        return None

    if isinstance(item, (tuple, list)):
        try:
            if len(item) == 3:
                s, e, lab = float(item[0]), float(item[1]), str(item[2])
                src = "manual"
            elif len(item) >= 4:
                s, e, lab, src = float(item[0]), float(item[1]), str(item[2]), str(item[3])
            else:
                return None
        except (ValueError, TypeError):
            return None
    elif isinstance(item, dict):
        try:
            s = float(item.get("start", item.get("s", None)))
            e = float(item.get("end", item.get("e", None)))
            lab = str(item.get("label", item.get("text", "")))
            src = str(item.get("source", item.get("src", "manual")))
        except (ValueError, TypeError):
            return None
    else:
        return None

    if e <= s:
        return None
    if not lab.strip():
        return None

    if target_source is not None:
        src = target_source

    return {"start": s, "end": e, "label": lab.strip(), "source": src.strip() or "manual"}


# ---------------------------------------------------------------------------
# 2. parse_annotation_row
# ---------------------------------------------------------------------------

def parse_annotation_row(
    parts: Sequence[str],
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Convert one row (list of string fields) into a standard annotation dict
    using the 1-based column mapping in *config*.

    Returns None when the row is unparseable, the label is empty, or
    start/end are not valid floats.
    """
    cfg = _build_config(config)

    def _col(key: str, default: int) -> int:
        try:
            return int(cfg.get(key, default)) - 1
        except (ValueError, TypeError):
            return int(default) - 1

    i_s = _col("start_col", 1)
    i_e = _col("end_col", 2)
    i_l = _col("label_col", 3)

    try:
        src_col = int(cfg.get("source_col", 4))
    except (ValueError, TypeError):
        src_col = 4
    i_src = src_col - 1

    if not isinstance(parts, (list, tuple)):
        return None
    max_needed = max(i_s, i_e, i_l)
    if max_needed < 0 or max_needed >= len(parts):
        return None

    try:
        s = float(str(parts[i_s]).strip())
        e = float(str(parts[i_e]).strip())
    except (ValueError, TypeError):
        return None

    lab = str(parts[i_l]).strip()
    if not lab:
        return None
    if e <= s:
        return None

    src = "manual"
    if 0 <= i_src < len(parts):
        src0 = str(parts[i_src]).strip()
        if src0:
            src = src0

    return {"start": s, "end": e, "label": lab, "source": src}


# ---------------------------------------------------------------------------
# 3–5. read_annotations_csv / txt / json
# ---------------------------------------------------------------------------

def _detect_delimiter(
    line: str,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Resolve the delimiter character for a csv/txt line."""
    cfg = _build_config(config)
    delim = str(cfg.get("delimiter", "auto")).strip().lower()

    if delim in {"comma", ","}:
        return ","
    if delim in {"semicolon", ";"}:
        return ";"
    if delim in {"tab", "\\t"}:
        return "\t"
    if delim in {"space", "whitespace"}:
        return " "
    if delim == "custom":
        custom = str(cfg.get("custom_delimiter", ""))
        return custom if custom else " "

    # auto
    if "," in line:
        return ","
    if "\t" in line:
        return "\t"
    if ";" in line:
        return ";"
    return " "


def _split_line(
    line: str,
    config: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Split a single csv/txt line into fields per config."""
    delimiter = _detect_delimiter(line, config)
    if delimiter == " ":
        return line.split()
    try:
        return [p.strip() for p in next(csv.reader([line], delimiter=delimiter))]
    except Exception:
        return [p.strip() for p in line.split(delimiter)]


def _read_text_flexible(path: str) -> str:
    """Read text file with fallback encodings (utf-8-sig, utf-8, gbk)."""
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def read_annotations_csv(
    path: str,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Read annotations from a CSV file.

    Returns a list of standard annotation dicts.  Malformed or skipped rows
    are silently dropped (matching legacy behaviour).
    """
    return _read_annotations_text(path, config)


def read_annotations_txt(
    path: str,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Read annotations from a TXT file (delimited text).

    Returns a list of standard annotation dicts.
    """
    return _read_annotations_text(path, config)


def _read_annotations_text(
    path: str,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    cfg = _build_config(config)
    content = _read_text_flexible(path)
    if not content:
        return []

    try:
        skip_n = int(cfg.get("skip_header_lines", 0))
    except (ValueError, TypeError):
        skip_n = 0

    rows: List[Dict[str, Any]] = []
    for line_idx, raw_line in enumerate(content.splitlines()):
        if line_idx < skip_n:
            continue
        line = raw_line.strip()
        if not line:
            continue
        parts = _split_line(line, config)
        ann = parse_annotation_row(parts, config)
        if ann is None:
            continue
        rows.append(ann)
    return rows


# ---- JSON helpers -------------------------------------------------------

def _annotation_row_from_dict(
    item: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Parse a JSON dict into a standard annotation dict.

    Key names are looked up from *config* (json_start_key, json_end_key,
    json_label_key, json_source_key) with common fallback synonyms.
    """
    cfg = _build_config(config)
    if not isinstance(item, dict):
        return None

    def _get_by_keys(primary: str, fallbacks: Sequence[str]) -> Any:
        keys = [primary] + list(fallbacks)
        for k in keys:
            if k in item:
                return item.get(k)
        lower_map = {str(k).lower(): k for k in item.keys()}
        for k in keys:
            kk = lower_map.get(str(k).lower())
            if kk is not None:
                return item.get(kk)
        return None

    start_v = _get_by_keys(
        str(cfg.get("json_start_key", "start")),
        ["start_time", "start_sec", "onset", "begin", "s"],
    )
    end_v = _get_by_keys(
        str(cfg.get("json_end_key", "end")),
        ["end_time", "end_sec", "offset", "finish", "e"],
    )
    label_v = _get_by_keys(
        str(cfg.get("json_label_key", "label")),
        ["text", "tag", "type", "class", "name"],
    )
    source_v = _get_by_keys(
        str(cfg.get("json_source_key", "source")),
        ["src", "origin"],
    )

    try:
        s = float(start_v)
        e = float(end_v)
    except (ValueError, TypeError):
        return None

    lab = str(label_v).strip() if label_v is not None else ""
    if not lab:
        return None
    if e <= s:
        return None

    src = str(source_v).strip() if source_v is not None and str(source_v).strip() else "manual"
    return {"start": s, "end": e, "label": lab, "source": src}


def _flatten_json_records(obj: Any, config: Optional[Dict[str, Any]] = None) -> List[Any]:
    """Extract annotation records from a JSON tree.

    Handles:
    - top-level list of dicts / lists
    - nested dict with keys "annotations" / "events" / "labels" / "data" / "items" / "records"
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # try self-as-record
        if _annotation_row_from_dict(obj, config) is not None:
            return [obj]
        for key in ("annotations", "events", "labels", "data", "items", "records"):
            val = obj.get(key)
            if isinstance(val, list):
                return val
    return []


def read_annotations_json(
    path: str,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Read annotations from a JSON file.

    Supports:
    - Top-level array of dicts
    - Wrapped ``{"annotations": [...]}`` style (or events / labels / data / items / records)
    - Entries can be dicts (keys from config) or arrays (columns from config)
    """
    content = _read_text_flexible(path)
    if not content:
        return []

    try:
        obj = json.loads(content)
    except Exception:
        return []

    rows: List[Dict[str, Any]] = []
    for rec in _flatten_json_records(obj, config):
        if isinstance(rec, dict):
            ann = _annotation_row_from_dict(rec, config)
        elif isinstance(rec, (list, tuple)):
            ann = parse_annotation_row(list(rec), config)
        else:
            ann = None
        if ann is not None:
            rows.append(ann)
    return rows


# ---------------------------------------------------------------------------
# 6. read_annotations (auto-detect format)
# ---------------------------------------------------------------------------

def read_annotations(
    path: str,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Read annotations from *path*, auto-detecting csv / txt / json.

    If the file format is forced in *config* (``file_format`` key), that
    takes precedence over extension sniffing.
    """
    cfg = _build_config(config)
    fmt = str(cfg.get("file_format", "auto")).strip().lower()
    ext = os.path.splitext(str(path))[1].lower().lstrip(".")
    if fmt == "auto":
        fmt = ext if ext in {"csv", "txt", "json"} else "txt"

    if fmt == "json":
        return read_annotations_json(path, config)
    # csv / txt share the same parser
    return _read_annotations_text(path, config)


# ---------------------------------------------------------------------------
# 7. write_annotations_csv
# ---------------------------------------------------------------------------

def write_annotations_csv(
    path: str,
    annotations: List[Dict[str, Any]],
) -> None:
    """Write annotations to a CSV file (start, end, label, source columns).

    Skips items with ``source == "archived"`` (matching legacy export behaviour).
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "label", "source"])
        for ann in annotations:
            if ann is None:
                continue
            src = str(ann.get("source", "manual"))
            if src.strip().lower() == "archived":
                continue
            w.writerow([
                f"{float(ann['start']):.4f}",
                f"{float(ann['end']):.4f}",
                str(ann["label"]),
                src,
            ])


# ---------------------------------------------------------------------------
# 8. write_annotations_json
# ---------------------------------------------------------------------------

def write_annotations_json(
    path: str,
    annotations: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Write annotations as a JSON array under the ``"annotations"`` key.

    If *metadata* is provided it is merged at the top level.
    """
    out: Dict[str, Any] = {"annotations": []}
    if metadata:
        out.update(metadata)

    for ann in annotations:
        if ann is None:
            continue
        src = str(ann.get("source", "manual"))
        if src.strip().lower() == "archived":
            continue
        out["annotations"].append({
            "start": float(ann["start"]),
            "end": float(ann["end"]),
            "label": str(ann["label"]),
            "source": src,
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 9. write_annotations (auto-detect CSV / JSON)
# ---------------------------------------------------------------------------

def write_annotations(
    path: str,
    annotations: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Write annotations, choosing format from file extension.

    - ``.json`` → JSON
    - anything else → CSV
    """
    ext = os.path.splitext(str(path))[1].lower().lstrip(".")
    if ext == "json":
        write_annotations_json(path, annotations, metadata)
    else:
        write_annotations_csv(path, annotations)


# ---------------------------------------------------------------------------
# 10. roundtrip_annotations
# ---------------------------------------------------------------------------

def roundtrip_annotations(
    path: str,
    annotations: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Write *annotations* to *path*, then read them back.

    Useful as a smoke-test: the returned list should contain the same
    annotation data (modulo float rounding).
    """
    write_annotations(path, annotations)
    return read_annotations(path, config)
