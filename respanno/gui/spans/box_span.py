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
                         # 默认“只读”：不允许拖动/改边界；仅在编辑模式下开启
                         movable=False,
                         resizable=False,  # 禁掉四角把手
                         pen=pg.mkPen(255, 255, 255, 255, width=1))
        # —— 视觉编码：不再强制纯白填充 (后面统一由 _apply_visual_style() 控制)——
        try:
            self.setBrush(pg.mkBrush(0, 0, 0, 0))  # 先透明，避免白块遮挡
        except Exception:
            pass

        self.owner = owner
        self.y_base = float(y_base)  # 锁定纵向
        self.h_fix = float(height)  # 锁定高度
        self.text = text
        self.setZValue(6)
        self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)

        # 独立填充层：RectROI 在部分 pyqtgraph 版本中只明显Display边框，
        # 因此额外用一个 QGraphicsRectItem 负责稳定Display类别填充色。
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

        # —— 编辑模式门控 (默认关闭，双击进入)——
        self._edit_mode = False
        self._orig_interval = None  # (x0, x1)
        self._orig_item = None      # data层快照 (用于编辑提交后可撤销)
        self._handles = []

        # ★ 新增：标签文字颜色 (不传则默认黑色)
        if label_color is None:
            self.label_color = QColor(0, 0, 0)  # 默认黑字
        else:
            self.label_color = label_color

        # 只保留左右把手 (默认隐藏+禁用；仅在编辑模式启用)
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
            # 禁用把手的鼠标交互 (避免与“标记拖拽”冲突)
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

        # 备注小白框 (Display在条上方)
        self.label = SpanLabelItem(owner_span=self, anchor=(0.5, 0.0))  # 先建一个空的
        self.owner.annot_plot.addItem(self.label)
        try:
            self.label._owner_span = self
        except Exception:
            pass
        self._update_label_html()  # 根据 label_color & text 更新 HTML
        # 标注条不直接Display文字，标签含义统一通过工具栏“Annotation Legend”查看
        try:
            self.label.hide()
        except Exception:
            pass

        # —— 视觉编码：根据 source(manual/ml) 统一Settings pen/brush/label 样式 ——
        self._apply_visual_style(self._infer_source())

        self.sigRegionChanged.connect(self._on_changed)
        self._on_changed()

    # ==========================
    # 编辑模式：仅在该模式下允许拖动/改边界
    # ==========================
    def enter_edit_mode(self):
        if self._edit_mode:
            return
        self._edit_mode = True
        self._orig_interval = self.interval()

        # 记录 data 层快照 (用于提交后可撤销)
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

        # 允许整体平移
        try:
            self.setMovable(True)
        except Exception:
            pass

        # 启用并Display左右把手 (仅用于水平改边界)
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

        # 视觉Notice：编辑态用虚线 (不改变 source 的语义；Exit后恢复)
        try:
            pen_now = getattr(self, "pen", None)
        except Exception:
            pen_now = None
        try:
            base = self.owner.get_annotation_color(getattr(self, "text", ""))
            edge_color = QColor(base)
            edge_color.setAlpha(230)
            self.setPen(pg.mkPen(edge_color, width=2.5, style=Qt.DashDotLine))
        except Exception:
            pass

        # 通知 owner：进入编辑态 (用于键盘 Enter/Esc)
        try:
            self.owner.begin_edit_span(self)
        except Exception:
            pass

        # 初次刷新状态栏
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

        # 取消：恢复原区间 (不入 undo)
        if not commit and old_interval is not None:
            try:
                x0, x1 = old_interval
                self.setPos([float(x0), self.y_base], update=False)
                self.setSize([float(x1 - x0), self.h_fix], update=False)
            except Exception:
                pass

        self._edit_mode = False

        # 禁用移动
        try:
            self.setMovable(False)
        except Exception:
            pass

        # 隐藏并禁用把手
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

        # 强制同步一次 (更新 annotations/spec/label + 恢复 source 对应样式)
        try:
            self._on_changed()
        except Exception:
            pass

        # 提交：若区间发生变化，则入栈 undo，支持 Ctrl+Z
        if commit and idx is not None and old_item is not None and old_interval is not None:
            try:
                n0, n1 = self.interval()
                o0, o1 = float(old_interval[0]), float(old_interval[1])
                if abs(n0 - o0) > 1e-9 or abs(n1 - o1) > 1e-9:
                    # new_item 以 data 层为准
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
                            # 若编辑的是机器/认可标注，则将 source 升级为 auto_edited (便于进入训练与导出追踪)
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

        # 清理快照
        self._orig_interval = None
        self._orig_item = None

        try:
            self.owner.end_edit_span(self)
        except Exception:
            pass

    # ==========================
    # 视觉编码：新增的辅助函数
    # ==========================
    def _infer_source(self) -> str:
        """
        从 owner.annotations 里推断当前 span 的 source。
        找不到就默认 manual。仅用于视觉编码，不改变逻辑。
        """
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
        """
        只做视觉编码：
        - 颜色 (边框/填充)：来自标签类别颜色 (owner.get_annotation_color)，避免人工标记编辑后边框变黑/消失
        - source：只决定线型/透明度 (manual 更“权威”，未确认机器标注更“建议”)
        """
        # Box 基色：按标签类别稳定映射 (不要用 label_color；label_color 仅用于文字颜色)
        try:
            base = self.owner.get_annotation_color(getattr(self, "text", ""))
            c = QColor(base)
        except Exception:
            c = QColor(255, 255, 255)

        # 主色 (边框/强调)
        edge_color = QColor(c)
        try:
            edge_color.setAlpha(220)
        except Exception:
            pass

        src_n = str(src).strip().lower()

        # 视觉约定：
        # - 未确认的机器标注：虚线 (ml/auto/pred 等)或 merged_*global*
        # - 已认可/已编辑/局部合并后的标注：实线 (auto_accepted/auto_edited/merged 等)
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

        # 应用到 ROI 本体
        try:
            self.setPen(pen)
        except Exception:
            try:
                self.pen = pen
                self.update()
            except Exception:
                pass

        # RectROI 自身仍保留淡填充；独立填充层负责主要可见色块。
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

        # 同步 label 的视觉 (不改文字内容)
        self._update_label_html(label_bg=label_bg, label_border=label_border)

    def _sync_fill_item(self):
        """同步独立填充矩形的位置与大小。"""
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
        """
        根据当前的 self.text 和 self.label_color 更新 label 的 HTML。
        仅视觉编码：支持可选的背景透明度与边框颜色。
        """
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
        """
        外部修改文字颜色的统一入口。
        """
        self.label_color = color
        # 颜色变了，视觉也应一起刷新 (manual/ml 不变)
        self._apply_visual_style(self._infer_source())

    def interval(self):
        p = self.pos()
        s = self.size()
        return float(p.x()), float(p.x() + s.x())

    def _on_changed(self):
        # 锁定纵向 & 高度
        p = self.pos()
        s = self.size()
        if float(p.y()) != self.y_base:
            self.setPos([p.x(), self.y_base], update=False)
            p = self.pos()
        if float(s.y()) != self.h_fix:
            self.setSize([s.x(), self.h_fix], update=False)
            s = self.size()

        # 备注框跟随 (在条上方一点点)
        cx = float(p.x() + s.x() / 2.0)
        cy = float(self.y_base + s.y()) + 0.05
        self.label.setPos(cx, cy)
        self._sync_fill_item()

        # 同步频谱红条
        m = getattr(self.owner, "_span2spec", {})
        if self in m:
            a, b = self.interval()
            m[self].setRegion([a, b])

        # 同步导出缓存 (兼容 3/4 元组；三元组视为人工标注)
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

                # —— 视觉编码：
                # 非编辑态：按 source 同步样式
                # 编辑态：保持“编辑态虚线”Notice，不要被 source 样式覆盖
                if not getattr(self, "_edit_mode", False):
                    self._apply_visual_style(src)

        # 编辑态：实时在状态栏Display起止Time
        if getattr(self, "_edit_mode", False):
            try:
                self.owner._update_edit_status(self)
            except Exception:
                pass

    def mouseDoubleClickEvent(self, ev):
        # 双击切换编辑模式：默认进入；编辑态再双击提交Exit
        try:
            if not self._edit_mode:
                self.enter_edit_mode()
            else:
                self.exit_edit_mode(commit=True)
            ev.accept()
            return
        except Exception:
            pass
        # 兜底：如果出错，仍允许Play
        try:
            s0, s1 = self.interval()
            self.owner.open_loop_player(s0, s1)
        except Exception:
            pass
        ev.accept()

    def mouseDragEvent(self, ev):
        # 非编辑模式下：吞掉拖拽 (不移动/不改边界)，避免与“标记拖拽”冲突
        if not getattr(self, "_edit_mode", False):
            try:
                ev.accept()
            except Exception:
                pass
            return
        # 编辑模式下：交给 ROI 默认实现 (支持平移/改边界)
        return super().mouseDragEvent(ev)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.RightButton:
            menu = QMenu()

            # —— 机器标注“认可”：将 source 置为 auto_accepted (进入训练/计入已审阅前缀)——
            src = ""
            try:
                src = str(self._infer_source()).strip().lower()
            except Exception:
                src = ""
            ml_like = src in {"ml", "auto", "machine", "model", "pred"}

            accept_action = None
            if ml_like:
                accept_action = menu.addAction("✅ Accept (use for training)")

            play_action = menu.addAction("▶ Play")
            del_action = menu.addAction("🗑 Delete")

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

