"""Dialog for entering/editing annotation label text with built-in presets."""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLabel,
    QComboBox, QLineEdit, QDialogButtonBox,
)

class AnnotationLabelDialog(QDialog):
    """标注标签选择对话框：支持预设类型 + Custom文本"""

    def __init__(self, parent=None, builtin_labels=None, start=None, end=None, default_text=""):
        super().__init__(parent)
        self.setWindowTitle("Add Annotation")
        self._text = None

        layout = QVBoxLayout(self)

        # 顶部Notice：Time段
        if start is not None and end is not None:
            info = QLabel(f"Annotation interval: {start:.3f} - {end:.3f} s")
            layout.addWidget(info)

        form = QFormLayout()
        layout.addLayout(form)

        # 预设类型下拉框：如 哮鸣音(Wheeze)、爆裂音(Crackles) 等
        self.combo = QComboBox()
        self.combo.addItem("(No preset)", userData=None)
        if builtin_labels:
            for cn, en in builtin_labels:
                self.combo.addItem(f"{cn} ({en})", userData=en)
        form.addRow("Preset type:", self.combo)

        # 文本输入框：最终用于保存的标签文本 (通常是英文)
        self.line_edit = QLineEdit()
        if default_text:
            self.line_edit.setText(default_text)
        form.addRow("Label text:", self.line_edit)

        # 选择预设时，自动把英文名写入文本框
        def on_combo_changed(idx):
            en = self.combo.itemData(idx)
            if en:
                self.line_edit.setText(en)

        self.combo.currentIndexChanged.connect(on_combo_changed)

        # 底部按钮
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _on_accept(self):
        text = self.line_edit.text().strip()
        if not text:
            # 不输入则视为取消
            self.reject()
            return
        self._text = text
        self.accept()

    def get_text(self):
        """模态执行并返回最终文本 (可能为 None)。"""
        if self.exec_() == QDialog.Accepted:
            return self._text
        return None



