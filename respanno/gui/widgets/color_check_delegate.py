"""Custom delegate for rendering colored checkboxes in feature selection."""

from PyQt5.QtWidgets import QStyledItemDelegate, QStyleOptionButton, QApplication, QStyle
from PyQt5.QtGui import QPalette
from PyQt5.QtCore import Qt

class ColorCheckDelegate(QStyledItemDelegate):
    """勾选后才上色；未勾选Display空框；整行可点击切换；方块尺寸更大"""
    BOX_SIZE = 22  # 放大方块 (原来16)
    PADDING_X = 10

    def paint(self, painter, option, index):
        painter.save()
        rect = option.rect

        # 背景
        if option.state & QStyle.State_Selected:
            painter.fillRect(rect, option.palette.highlight())
        else:
            painter.fillRect(rect, option.palette.base())

        # 勾选状态
        checked = (index.data(Qt.CheckStateRole) == Qt.Checked)

        # 颜色 (来自 UserRole；未勾选也可以先拿着，但不填充)
        color = index.data(Qt.UserRole)
        if not isinstance(color, QColor):
            try:
                color = QColor(color) if color else QColor("#999999")
            except Exception:
                color = QColor("#999999")

        # 方块区域 (更大)
        box = QRect(
            rect.x() + self.PADDING_X,
            rect.y() + (rect.height() - self.BOX_SIZE) // 2,
            self.BOX_SIZE,
            self.BOX_SIZE
        )

        painter.setRenderHint(QPainter.Antialiasing, True)
        # 边框
        border_pen = QPen(color.darker(140) if checked else QColor(160, 160, 160), 1)
        painter.setPen(border_pen)
        # 填充：仅已勾选才填充彩色；未勾选Display空框
        painter.setBrush(QBrush(color) if checked else Qt.NoBrush)
        painter.drawRoundedRect(box, 4, 4)

        # 勾选对号：仅已勾选绘制
        if checked:
            painter.setPen(QPen(Qt.white, 2))
            path = QPainterPath()
            path.moveTo(box.left() + 4, box.center().y())
            path.lineTo(box.center().x() - 1, box.bottom() - 4)
            path.lineTo(box.right() - 4, box.top() + 5)
            painter.drawPath(path)

        # 文本
        text_rect = QRect(box.right() + 10, rect.y(), rect.width() - (box.width() + 20), rect.height())
        painter.setPen(option.palette.text().color())
        painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, index.data())

        painter.restore()

    def sizeHint(self, option, index):
        s = super().sizeHint(option, index)
        # 放大行高，增大点击目标
        return QSize(max(s.width(), 80), max(s.height(), 30))

    def editorEvent(self, event, model, option, index):
        """把整行都变成点击区域：点击行内任意处都切换勾选"""
        if event.type() in (QEvent.MouseButtonRelease, QEvent.MouseButtonDblClick):
            if option.rect.contains(event.pos()):
                state = index.data(Qt.CheckStateRole)
                new_state = Qt.Unchecked if state == Qt.Checked else Qt.Checked
                # 写回 CheckStateRole
                model.setData(index, new_state, Qt.CheckStateRole)
                return True
        return super().editorEvent(event, model, option, index)


from respanno.gui.widgets.clickable_slider import ClickableSlider  # noqa: F401

