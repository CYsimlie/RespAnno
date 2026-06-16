"""Core annotation visual element: a rectangular span with label text.

BoxSpan is a pg.RectROI subclass representing one annotation on the timeline.
It handles visual styling (color by source), edit mode, label display, and
right-click context menus.

Imports SpanLabelItem from the spans subpackage.
"""

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QMenu, QAction, QGraphicsRectItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPen, QBrush

from respanno.gui.spans.span_label_item import SpanLabelItem

class BoxSpan(pg.RectROI):
    def __init__(self, x0, x1, y_base, height, text, owner, label_color=None):
        super().__init__(pos=[x0, y_base], size=[x1 - x0, height],
                         movable=False,
                         resizable=False,
                         pen=pg.mkPen(255, 255, 255, 255, width=1))
        self.setBrush(pg.mkBrush(0, 0, 0, 0))

        self.owner = owner
        self.y_base = float(y_base)
        self.h_fix = float(height)
        self.text = text
        self.setZValue(6)
        self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)

        # Separate fill layer for stable category color display via QGraphicsRectItem.
        self._fill_item = None
        try:
            self._fill_item = QGraphicsRectItem(float(x0), float(y_base), float(x1 - x0), float(height))
            self._fill_item.setPen(QPen(Qt.NoPen))
            self._fill_item.setBrush(QBrush(QColor(255, 255, 255, 80)))
            self._fill_item.setZValue(self.zValue() - 1)
            self._fill_item.setAcceptedMouseButtons(Qt.NoButton)
            self.owner.annot_plot.addItem(self._fill_item)
        except Exception:
            self._fill_item = None

        # Double-click to enter edit mode.
        self._edit_mode = False
        self._orig_interval = None
        self._orig_item = None
        self._handles = []

        if label_color is None:
            self.label_color = QColor(0, 0, 0)
        else:
            self.label_color = label_color

        # Side handles are hidden by default; visible only in edit mode.
        hL = self.addScaleHandle([0, 0.5], [1, 0.5])
        hR = self.addScaleHandle([1, 0.5], [0, 0.5])
        self._handles = [hL, hR]

        def _hide_handle(h):
            try:
                h.setOpacity(0.0)
                h.setPen(pg.mkPen(0, 0, 0, 0))
                h.setBrush(pg.mkBrush(0, 0, 0, 0))
                return
            except Exception:
                pass
            try:
                it = h['item']
                it.setOpacity(0.0)
                it.setPen(pg.mkPen(0, 0, 0, 0))
                it.setBrush(pg.mkBrush(0, 0, 0, 0))
            except Exception:
                pass

        for h in (hL, hR):
            _hide_handle(h)
            # Hide mouse interaction to avoid conflicts with drag-to-mark.
            try:
                h.setAcceptedMouseButtons(Qt.NoButton)
            except Exception:
                try:
                    it = h['item']
                    it.setAcceptedMouseButtons(Qt.NoButton)
                except Exception:
                    pass

        def _style_handle(h):
            try:
                h.setPen(pg.mkPen(255, 255, 255, 220))
                h.setBrush(pg.mkBrush(0, 0, 0, 160))
                h.setZValue(self.zValue() + 1)
                return
            except Exception:
                pass
            try:
                it = h['item']
                it.setPen(pg.mkPen(255, 255, 255, 220))
                it.setBrush(pg.mkBrush(0, 0, 0, 160))
                it.setZValue(self.zValue() + 1)
            except Exception:
                pass

        for h in (hL, hR):
            _style_handle(h)

        self.label = SpanLabelItem(owner_span=self, anchor=(0.5, 0.0))
        self.owner.annot_plot.addItem(self.label)
        try:
            self.label._owner_span = self
        except Exception:
            pass
        self._update_label_html()
        try:
            self.label.hide()
        except Exception:
            pass

        self._apply_visual_style(self._infer_source())

        self.sigRegionChanged.connect(self._on_changed)
        self._on_changed()

    def enter_edit_mode(self):
        if self._edit_mode:
            return
        self._edit_mode = True
        self._orig_interval = self.interval()

        # Save data-layer snapshot for undo.
        self._orig_item = None
        try:
            idx = None
            try:
                idx = int(getattr(self.owner, "_span2idx", {}).get(self))
            except Exception:
                idx = None
            if idx is not None and 0 <= idx < len(getattr(self.owner, "annotations", [])):
                it = self.owner.annotations[idx]
                if it is not None:
                    if len(it) == 3:
                        self._orig_item = (float(it[0]), float(it[1]), str(it[2]), "manual")
                    else:
                        self._orig_item = (float(it[0]), float(it[1]), str(it[2]), str(it[3]) if len(it) >= 4 else "manual")
        except Exception:
            self._orig_item = None

        # Allow translation in edit mode.
        try:
            self.setMovable(True)
        except Exception:
            pass

        for h in getattr(self, "_handles", []):
            try:
                h.setOpacity(1.0)
                h.setAcceptedMouseButtons(Qt.LeftButton)
            except Exception:
                try:
                    it = h['item']
                    it.setOpacity(1.0)
                    it.setAcceptedMouseButtons(Qt.LeftButton)
                except Exception:
                    pass

        # Dashed border in edit mode.
        try:
            base = self.owner.get_annotation_color(getattr(self, "text", ""))
            edge_color = QColor(base)
            edge_color.setAlpha(230)
            self.setPen(pg.mkPen(edge_color, width=2.5, style=Qt.DashDotLine))
        except Exception:
            pass

        try:
            self.owner.begin_edit_span(self)
        except Exception:
            pass

        try:
            self.owner._update_edit_status(self)
        except Exception:
            pass

    def exit_edit_mode(self, commit: bool = True):
        if not self._edit_mode:
            return

        old_interval = self._orig_interval
        old_item = getattr(self, "_orig_item", None)

        idx = None
        try:
            idx = int(getattr(self.owner, "_span2idx", {}).get(self))
        except Exception:
            idx = None

        # Cancel: restore original interval (no undo push).
        if not commit and old_interval is not None:
            try:
                x0, x1 = old_interval
                self.setPos([float(x0), self.y_base], update=False)
                self.setSize([float(x1 - x0), self.h_fix], update=False)
            except Exception:
                pass

        self._edit_mode = False

        # Disable translation.
        try:
            self.setMovable(False)
        except Exception:
            pass

        for h in getattr(self, "_handles", []):
            try:
                h.setOpacity(0.0)
                h.setAcceptedMouseButtons(Qt.NoButton)
            except Exception:
                try:
                    it = h['item']
                    it.setOpacity(0.0)
                    it.setAcceptedMouseButtons(Qt.NoButton)
                except Exception:
                    pass

        # Sync to data layer and restore visual style.
        try:
            self._on_changed()
        except Exception:
            pass

        # Commit: push undo record if interval changed.
        if commit and idx is not None and old_item is not None and old_interval is not None:
            try:
                n0, n1 = self.interval()
                o0, o1 = float(old_interval[0]), float(old_interval[1])
                if abs(n0 - o0) > 1e-9 or abs(n1 - o1) > 1e-9:
                    new_item = None
                    try:
                        if 0 <= idx < len(self.owner.annotations):
                            it = self.owner.annotations[idx]
                            if it is not None:
                                if len(it) == 3:
                                    new_item = (float(it[0]), float(it[1]), str(it[2]), "manual")
                                else:
                                    new_item = (float(it[0]), float(it[1]), str(it[2]), str(it[3]) if len(it) >= 4 else "manual")
                    except Exception:
                        new_item = None

                    if new_item is not None:
                        try:
                            try:
                                _old_src = str(old_item[3]) if (old_item is not None and len(old_item) >= 4) else "manual"
                            except Exception:
                                _old_src = "manual"
                            # If editing a machine or accepted annotation, upgrade source to auto_edited.
                            try:
                                _old_src_n = str(_old_src).strip().lower()
                                if _old_src_n in {"ml", "auto", "machine", "model", "pred", "auto_accepted"}:
                                    new_item = (float(new_item[0]), float(new_item[1]), str(new_item[2]), "auto_edited")
                                    try:
                                        if 0 <= idx < len(self.owner.annotations):
                                            self.owner.annotations[idx] = new_item
                                    except Exception:
                                        pass
                                    try:
                                        self._apply_visual_style("auto_edited")
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                            try:
                                _new_src = str(new_item[3]) if (new_item is not None and len(new_item) >= 4) else "manual"
                            except Exception:
                                _new_src = "manual"
                            try:
                                print(f"[VERIFY][EDIT] idx={idx} label={new_item[2] if new_item is not None else ''} "
                                      f"src_old={_old_src} src_new={_new_src} "
                                      f"{o0:.4f}-{o1:.4f} -> {n0:.4f}-{n1:.4f}")
                            except Exception:
                                pass

                            self.owner._push_undo({
                                "op": "edit_interval",
                                "idx": int(idx),
                                "old_item": old_item,
                                "new_item": new_item,
                            })
                        except Exception:
                            pass
            except Exception:
                pass

        # Clean up snapshots
        self._orig_interval = None
        self._orig_item = None

        try:
            self.owner.end_edit_span(self)
        except Exception:
            pass

    def _infer_source(self) -> str:
        """Infer source provenance from owner.annotations; fallback to manual (visual-only, no logic change)."""
        try:
            m2 = getattr(self.owner, "_span2idx", {})
            if self in m2:
                idx = m2[self]
                if 0 <= idx < len(self.owner.annotations) and self.owner.annotations[idx] is not None:
                    old = self.owner.annotations[idx]
                    if len(old) >= 4:
                        return str(old[3])
        except Exception:
            pass
        return "manual"

    def _apply_visual_style(self, src: str):
        """Apply visual style: color from label category, line/opacity from source provenance."""
        try:
            base = self.owner.get_annotation_color(getattr(self, "text", ""))
            c = QColor(base)
        except Exception:
            c = QColor(255, 255, 255)

        # Primary color (border / emphasis)
        edge_color = QColor(c)
        try:
            edge_color.setAlpha(220)
        except Exception:
            pass

        src_n = str(src).strip().lower()

        # Visual conventions:
        #   Unconfirmed ML annotations: dashed border.
        #   Accepted / edited / merged annotations: solid border.
        is_ml_like = (
            src_n in {"ml", "auto", "machine", "model", "pred"}
            or (src_n.startswith("merged") and ("global" in src_n))
        )

        if is_ml_like:
            pen = pg.mkPen(edge_color, width=1.5, style=Qt.DashLine)
            fill = QColor(c)
            try:
                fill.setAlpha(115)
            except Exception:
                pass
            label_bg = "rgba(255,255,255,0.80)"
            label_border = "rgba({},{},{},0.55)".format(c.red(), c.green(), c.blue())
        else:
            pen = pg.mkPen(edge_color, width=2.0, style=Qt.SolidLine)
            fill = QColor(c)
            try:
                fill.setAlpha(135)
            except Exception:
                pass
            label_bg = "rgba(255,255,255,0.90)"
            label_border = "rgba({},{},{},0.80)".format(c.red(), c.green(), c.blue())

        # Apply to ROI body
        try:
            self.setPen(pen)
        except Exception:
            try:
                self.pen = pen
                self.update()
            except Exception:
                pass

        # RectROI keeps faint fill; the independent fill layer handles the main colour block.
        try:
            self.setBrush(pg.mkBrush(fill))
        except Exception:
            pass
        try:
            fill_item = getattr(self, "_fill_item", None)
            if fill_item is not None:
                fill_item.setBrush(QBrush(fill))
                fill_item.setPen(QPen(Qt.NoPen))
                fill_item.show()
                self._sync_fill_item()
        except Exception:
            pass

        self._update_label_html(label_bg=label_bg, label_border=label_border)

    def _sync_fill_item(self):
        """Sync fill rectangle position and size."""
        try:
            fill_item = getattr(self, "_fill_item", None)
            if fill_item is None:
                return
            a, b = self.interval()
            fill_item.setRect(float(a), float(self.y_base), max(0.0, float(b - a)), float(self.h_fix))
            fill_item.setZValue(self.zValue() - 1)
        except Exception:
            pass

    def _update_label_html(self, label_bg: str = None, label_border: str = None):
        """Build and apply label HTML from self.text and self.label_color."""
        c = self.label_color
        color_str = "#{:02x}{:02x}{:02x}".format(c.red(), c.green(), c.blue())

        if label_bg is None:
            label_bg = "rgba(255,255,255,0.90)"
        if label_border is None:
            label_border = "rgba(120,120,120,0.85)"

        html = (
            f'<div style="background:{label_bg};'
            f'color:{color_str};'
            f'border:1px solid {label_border};border-radius:4px;'
            'padding:1px 4px;white-space:nowrap;font-size:11px">'
            f'{self.text}'
            '</div>'
        )
        try:
            self.label.setHtml(html)
        except Exception:
            pass

    def set_label_color(self, color: QColor):
        """Set label text colour and refresh visual style."""
        self.label_color = color
        self._apply_visual_style(self._infer_source())

    def interval(self):
        p = self.pos()
        s = self.size()
        return float(p.x()), float(p.x() + s.x())

    def _on_changed(self):
        # Lock vertical position and height.
        p = self.pos()
        s = self.size()
        if float(p.y()) != self.y_base:
            self.setPos([p.x(), self.y_base], update=False)
            p = self.pos()
        if float(s.y()) != self.h_fix:
            self.setSize([s.x(), self.h_fix], update=False)
            s = self.size()

        # Position the label slightly above the bar.
        cx = float(p.x() + s.x() / 2.0)
        cy = float(self.y_base + s.y()) + 0.05
        self.label.setPos(cx, cy)
        self._sync_fill_item()

        # Sync spectrum highlight bar.
        m = getattr(self.owner, "_span2spec", {})
        if self in m:
            a, b = self.interval()
            m[self].setRegion([a, b])

        # Sync annotations export cache (3-tuples treated as manual).
        m2 = getattr(self.owner, "_span2idx", {})
        if self in m2:
            idx = m2[self]
            if 0 <= idx < len(self.owner.annotations) and self.owner.annotations[idx] is not None:
                a, b = self.interval()
                s, e = a, b
                old = self.owner.annotations[idx]
                try:
                    if len(old) == 3:
                        _, _, t = old
                        src = "manual"
                    elif len(old) >= 4:
                        _, _, t, src = old[:4]
                    else:
                        t = self.text
                        src = "manual"
                except Exception:
                    t = self.text
                    src = "manual"
                self.owner.annotations[idx] = (s, e, t, src)

                # Non-edit: sync visual style from source.
                # Edit mode: keep dashed edit-mode hint.
                if not getattr(self, "_edit_mode", False):
                    self._apply_visual_style(src)

        # Show start/end times in status bar during edit.
        if getattr(self, "_edit_mode", False):
            try:
                self.owner._update_edit_status(self)
            except Exception:
                pass

    def mouseDoubleClickEvent(self, ev):
        # Toggle edit mode on double-click.
        try:
            if not self._edit_mode:
                self.enter_edit_mode()
            else:
                self.exit_edit_mode(commit=True)
            ev.accept()
            return
        except Exception:
            pass
        # Fallback: play the segment if edit toggle fails.
        try:
            s0, s1 = self.interval()
            self.owner.open_loop_player(s0, s1)
        except Exception:
            pass
        ev.accept()

    def mouseDragEvent(self, ev):
        # Non-edit mode: swallow drag events to avoid conflict with drag-to-mark.
        if not getattr(self, "_edit_mode", False):
            try:
                ev.accept()
            except Exception:
                pass
            return
        # Edit mode: delegate to ROI default (translation/resize).
        return super().mouseDragEvent(ev)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.RightButton:
            menu = QMenu()

            # Accept ML annotation: set source to auto_accepted.
            src = ""
            try:
                src = str(self._infer_source()).strip().lower()
            except Exception:
                src = ""
            ml_like = src in {"ml", "auto", "machine", "model", "pred"}

            accept_action = None
            if ml_like:
                accept_action = menu.addAction("Accept (use for training)")

            play_action = menu.addAction("Play")
            del_action = menu.addAction("Delete")

            act = menu.exec_(ev.screenPos().toPoint())

            if accept_action is not None and act == accept_action:
                try:
                    self.owner.accept_annotation(self, accepted_source="auto_accepted")
                except Exception:
                    pass
            elif act == play_action:
                s0, s1 = self.interval()
                self.owner.open_loop_player(s0, s1)
            elif act == del_action:
                self.owner.delete_annotation(self)
            ev.accept()
        else:
            super().mouseClickEvent(ev)

    def cleanup(self):
        try:
            if getattr(self, "_fill_item", None) is not None:
                self.owner.annot_plot.removeItem(self._fill_item)
                self._fill_item = None
        except Exception:
            pass
        try:
            self.owner.annot_plot.removeItem(self)
        except:
            pass
        try:
            self.owner.annot_plot.removeItem(self.label)
        except:
            pass



from respanno.gui.views.annot_view_box import AnnotViewBox  # noqa: F401
from respanno.gui.views.wave_view_box import WaveViewBox  # noqa: F401

