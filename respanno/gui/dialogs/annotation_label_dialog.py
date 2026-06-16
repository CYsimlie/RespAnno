"""Dialog for entering/editing annotation label text with built-in presets."""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLabel,
    QComboBox, QLineEdit, QDialogButtonBox,
)

class AnnotationLabelDialog(QDialog):
    """Annotation label dialog with preset types and custom text entry."""

    def __init__(self, parent=None, builtin_labels=None, start=None, end=None, default_text=""):
        super().__init__(parent)
        self.setWindowTitle("Add Annotation")
        self._text = None

        layout = QVBoxLayout(self)

        # Time-segment label.
        if start is not None and end is not None:
            info = QLabel(f"Annotation interval: {start:.3f} - {end:.3f} s")
            layout.addWidget(info)

        form = QFormLayout()
        layout.addLayout(form)

        # Preset label combo box (e.g., Wheeze, Crackles).
        self.combo = QComboBox()
        self.combo.addItem("(No preset)", userData=None)
        if builtin_labels:
            for cn, en in builtin_labels:
                self.combo.addItem(f"{cn} ({en})", userData=en)
        form.addRow("Preset type:", self.combo)

        # Text input for the final label.
        self.line_edit = QLineEdit()
        if default_text:
            self.line_edit.setText(default_text)
        form.addRow("Label text:", self.line_edit)

        # When a preset is selected, auto-fill the English name.
        def on_combo_changed(idx):
            en = self.combo.itemData(idx)
            if en:
                self.line_edit.setText(en)

        self.combo.currentIndexChanged.connect(on_combo_changed)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _on_accept(self):
        text = self.line_edit.text().strip()
        if not text:
            # Empty input means cancel.
            self.reject()
            return
        self._text = text
        self.accept()

    def get_text(self):
        """Run the dialog modally and return the label text (or None)."""
        if self.exec_() == QDialog.Accepted:
            return self._text
        return None



