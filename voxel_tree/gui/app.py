"""app.py — QApplication bootstrap and dark-theme setup."""

from __future__ import annotations

import sys

from PySide6.QtCore import QtMsgType, qInstallMessageHandler
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication


def create_app() -> QApplication:
    app = QApplication.instance() or QApplication(sys.argv)
    _install_qt_message_filter()
    _apply_dark_palette(app)
    return app  # type: ignore[return-value]


def _install_qt_message_filter() -> None:
    """Suppress noisy Qt warnings about stylesheet parsing in our GUI."""

    def handler(mode: QtMsgType, context, message: str) -> None:
        # Qt can emit "Could not parse stylesheet of object ..." during startup.
        # These messages are usually harmless and clutter the console.
        if "Could not parse stylesheet of object" in message:
            return
        sys.__stderr__.write(message + "\n")

    qInstallMessageHandler(handler)


def _apply_dark_palette(app: QApplication) -> None:
    """Apply a uniform dark colour palette."""
    app.setStyle("Fusion")
    palette = QPalette()
    dark = QColor(26, 26, 26)
    mid = QColor(40, 40, 40)
    text = QColor(200, 200, 200)
    highlight = QColor(58, 90, 140)

    palette.setColor(QPalette.ColorRole.Window, dark)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, mid)
    palette.setColor(QPalette.ColorRole.AlternateBase, dark)
    palette.setColor(QPalette.ColorRole.ToolTipBase, dark)
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.Button, mid)
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Highlight, highlight)
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
