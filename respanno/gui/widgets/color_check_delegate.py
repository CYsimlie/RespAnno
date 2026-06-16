"""Custom delegate for rendering colored checkboxes in feature selection."""

from PyQt5.QtWidgets import QStyledItemDelegate, QStyleOptionButton, QApplication, QStyle
from PyQt5.QtGui import QPalette, QColor, QPainter, QPen, QBrush, QPainterPath
from PyQt5.QtCore import Qt, QRect, QSize, QEvent

class ColorCheckDelegate(QStyledItemDelegate):
    """Colour-filled checkbox: full-row clickable, checked = filled, unchecked = hollow."""
    BOX_SIZE = 22
    PADDING_X = 10

    def paint(self, painter, option, index):
        painter.save()
        rect = option.rect

        # Background.
        if option.state & QStyle.State_Selected:
            painter.fillRect(rect, option.palette.highlight())
        else:
            painter.fillRect(rect, option.palette.base())

        # Check state.
        checked = (index.data(Qt.CheckStateRole) == Qt.Checked)

        # Colour from UserRole.
        color = index.data(Qt.UserRole)
        if not isinstance(color, QColor):
            try:
                color = QColor(color) if color else QColor("#999999")
            except Exception:
                color = QColor("#999999")

        # Box region.
        box = QRect(
            rect.x() + self.PADDING_X,
            rect.y() + (rect.height() - self.BOX_SIZE) // 2,
            self.BOX_SIZE,
            self.BOX_SIZE
        )

        painter.setRenderHint(QPainter.Antialiasing, True)
        # Border.
        border_pen = QPen(color.darker(140) if checked else QColor(160, 160, 160), 1)
        painter.setPen(border_pen)
        # Fill: only when checked.
        painter.setBrush(QBrush(color) if checked else Qt.NoBrush)
        painter.drawRoundedRect(box, 4, 4)

        # Checkmark: only when checked.
        if checked:
            painter.setPen(QPen(Qt.white, 2))
            path = QPainterPath()
            path.moveTo(box.left() + 4, box.center().y())
            path.lineTo(box.center().x() - 1, box.bottom() - 4)
            path.lineTo(box.right() - 4, box.top() + 5)
            painter.drawPath(path)

        # text
        text_rect = QRect(box.right() + 10, rect.y(), rect.width() - (box.width() + 20), rect.height())
        painter.setPen(option.palette.text().color())
        painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, index.data())

        painter.restore()

    def sizeHint(self, option, index):
        s = super().sizeHint(option, index)
        # Enlarged row for easier clicking.
        return QSize(max(s.width(), 80), max(s.height(), 30))

    def editorEvent(self, event, model, option, index):
        """Full-row clickable: click anywhere to toggle check state."""
        if event.type() in (QEvent.MouseButtonRelease, QEvent.MouseButtonDblClick):
            if option.rect.contains(event.pos()):
                state = index.data(Qt.CheckStateRole)
                new_state = Qt.Unchecked if state == Qt.Checked else Qt.Checked
                model.setData(index, new_state, Qt.CheckStateRole)
                return True
        return super().editorEvent(event, model, option, index)


from respanno.gui.widgets.clickable_slider import ClickableSlider  # noqa: F401

