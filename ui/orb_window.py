# orb_window.py
# A fully dynamic and animated UI for "Athena"

import os
import sys
import json
import random
import math
import subprocess
from pathlib import Path
from threading import Event

# === PyQt6 Core ===
from PyQt6.QtCore import (
    Qt, QTimer, QPoint, QRectF, QDir, QPointF, pyqtSignal, QEventLoop, QObject
)

# === PyQt6 GUI ===
from PyQt6.QtGui import (
    QAction, QPainter, QRadialGradient, QBrush, QPen, QColor,
    QContextMenuEvent, QRegion, QIcon, QMouseEvent
)
# === PyQt6 Widgets ===
from PyQt6.QtWidgets import (
    QWidget, QMenu, QMessageBox,
    QInputDialog, QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QDialogButtonBox,
    QFileDialog, QApplication
)

JSON_PATH = 'user_data/exec_paths.json'
CONTACT_PATH = 'user_data/contacts.json'

class OrbWindow(QWidget):
    clicked_signal = pyqtSignal(QPoint)
    def __init__(self):
        super().__init__()
        self._click_event = Event()
        self._click_pos   = QPoint()
        self.isOrbOpen = True
        self._quitting = False

        # ------------------- 1. Window setup -------------------
        self.orb_size = 150
        self.setFixedSize(self.orb_size, self.orb_size)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setToolTip("Athena")

        # ------------------- 2. Animation -------------------
        self.current_state = "idle"
        self.animation_frame = 0
        self.talk_h1 = self.talk_h2 = self.talk_h3 = 10.0

        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(16)               # ~60 fps

        self.talking_timer = QTimer(self)
        self.talking_timer.timeout.connect(self.update_talking_animation)
        self.talking_timer.setInterval(100)          # 10 fps

        self._set_circular_mask()

        # ------------------- 3. Mouse drag -------------------
        self.drag_pos = None

        # ------------------- 4. Load saved executables -------------------
        self.executables: dict[str, str] = {}        # name → full_path
        self._load_executables()

    # ------------------------------------------------------------------
    def _set_circular_mask(self):
        region = QRegion(self.rect(), QRegion.RegionType.Ellipse)
        self.setMask(region)

    # ------------------------------------------------------------------
    def set_state(self, state: str):
        if state not in {"idle", "listening", "talking", "processing"}:
            state = "idle"
        self.current_state = state
        self.talking_timer.start() if state == "talking" else self.talking_timer.stop()

    # ------------------------------------------------------------------
    def update_animation(self):
        self.animation_frame += 1
        self.update()

    # ------------------------------------------------------------------
    def update_talking_animation(self):
        self.talk_h1 = 10 + (random.randint(0, 100) / 100) * 45
        self.talk_h2 = 10 + (random.randint(0, 100) / 100) * 70
        self.talk_h3 = 10 + (random.randint(0, 100) / 100) * 45

    # ------------------------------------------------------------------
    # ------------------- Painting -------------------
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), Qt.GlobalColor.transparent)

        {"idle": self.draw_idle,
         "listening": self.draw_listening,
         "talking": self.draw_talking,
         "processing": self.draw_processing}[self.current_state](p)

    def draw_idle(self, p: QPainter):
        pulse = (math.sin(self.animation_frame * 0.05) + 1) / 2
        glow = self.orb_size * 0.7 + pulse * 10

        # FIX: Use QPointF
        center = self.rect().center().toPointF()
        grad = QRadialGradient(center, self.orb_size / 2)
        grad.setColorAt(0, QColor(0, 80, 150))
        grad.setColorAt(1, QColor(0, 20, 50))
        p.setBrush(grad); p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(self.rect())

        glow_c = QColor(0, 150, 255, int(30 + pulse * 30))
        p.setBrush(glow_c)
        p.drawEllipse(center, int(glow / 2), int(glow / 2))


    def draw_listening(self, p: QPainter):
        pulse = (math.sin(self.animation_frame * 0.15) + 1) / 2
        core = self.orb_size * 0.5 + pulse * 10

        center = self.rect().center().toPointF()
        grad = QRadialGradient(center, self.orb_size / 2)
        grad.setColorAt(0, QColor(0, 150, 255))
        grad.setColorAt(1, QColor(0, 50, 120))
        p.setBrush(grad); p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(self.rect())

        core_c = QColor(200, 230, 255, int(150 + pulse * 50))
        p.setBrush(core_c)
        p.drawEllipse(center, int(core / 2), int(core / 2))


    def draw_talking(self, p: QPainter):
        center = self.rect().center().toPointF()
        grad = QRadialGradient(center, self.orb_size / 2)
        grad.setColorAt(0, QColor(0, 120, 220))
        grad.setColorAt(1, QColor(0, 40, 100))
        p.setBrush(grad); p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(self.rect())

        w = 15
        cx, cy = self.rect().center().x(), self.rect().center().y()
        h1, h2, h3 = self.talk_h1, self.talk_h2, self.talk_h3
        p.setBrush(QColor(200, 230, 255))

        p.drawRect(QRectF(cx - w * 1.5, cy - h1 / 2, w, h1))
        p.drawRect(QRectF(cx - w * 0.5, cy - h2 / 2, w, h2))
        p.drawRect(QRectF(cx + w * 0.5, cy - h3 / 2, w, h3))


    def draw_processing(self, p: QPainter):
        center = self.rect().center().toPointF()
        grad = QRadialGradient(center, self.orb_size / 2)
        grad.setColorAt(0, QColor(130, 0, 180))
        grad.setColorAt(1, QColor(40, 0, 60))
        p.setBrush(grad); p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(self.rect())

        angle = (self.animation_frame * 3) % 360
        pen = QPen(QColor(230, 200, 255), 10)
        p.setPen(pen)
        p.drawArc(self.rect().adjusted(15, 15, -15, -15), angle * 16, 90 * 16)
    
    
    # ------------------- Mouse handling -------------------
    def mousePressEvent(self, ev: QMouseEvent):
        if ev.button() == Qt.MouseButton.LeftButton:
            self.drag_pos = ev.globalPosition().toPoint() - self.pos()
            self.clicked_signal.emit(ev.position().toPoint())

            ev.accept()
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QMouseEvent):
        if ev.buttons() == Qt.MouseButton.LeftButton and self.drag_pos:
            self.move(ev.globalPosition().toPoint() - self.drag_pos)
            ev.accept()

    def mouseReleaseEvent(self, ev: QMouseEvent):
        self.drag_pos = None
        ev.accept()

    # ------------------- Context menu -------------------
    def contextMenuEvent(self, ev: QContextMenuEvent):
        menu = QMenu(self)
        title = QAction("Athena", self)
        title.setEnabled(False)
        menu.addAction(title)
        menu.addSeparator()

        # --- Add Executable ---
        add_exe = QAction("Add Executable…", self)
        add_exe.triggered.connect(self._show_add_dialog)
        menu.addAction(add_exe)

        # --- NEW: Add Contact ---
        add_contact = QAction("Add Contact…", self)
        add_contact.triggered.connect(self._show_contact_dialog)
        menu.addAction(add_contact)

        # --- Quit ---
        quit_act = QAction("Quit Orb", self)  # ← Clearer label
        quit_act.triggered.connect(self._quit_gracefully)
        menu.addAction(quit_act)

        menu.exec(ev.globalPos())

    # ------------------------------------------------------------------
    def _show_add_dialog(self):
        dlg = ExecutableInputDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            path, keyword = dlg.get_data()
            self.executables[keyword] = path
            self._save_executables()
            QMessageBox.information(
                self, "Saved",
                f'Keyword "<b>{keyword}</b>" → <code>{path}</code>'
            )

   # ------------------- JSON persistence -------------------
    def _load_executables(self):
        """Load user-added executables from exec_paths.json (same folder)"""
        if not os.path.exists(JSON_PATH):
            return  # No file yet

        try:
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.executables = data
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", f"Error reading config:\n{e}")
    
    def _save_executables(self):
        """Save user-added executables to exec_paths.json (same folder)"""
        try:
            # No need to create folder — same as script, already exists
            with open(JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(self.executables, f, indent=2)
            print(f"Executables saved to: {JSON_PATH}")
        except Exception as e:
            QMessageBox.critical(
                self, "Save Failed",
                f"Could not save configuration:\n\n{e}\n\nLocation: {JSON_PATH}"
            )

    def _show_contact_dialog(self):
        dlg = ContactInputDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            name, email = dlg.get_data()
            self._save_contact(name, email)
            QMessageBox.information(
                self, "Saved",
                f'Contact "<b>{name}</b>" → <code>{email}</code>'
            )

    # ------------------------------------------------------------------
    def _save_contact(self, name: str, email: str):
        """Append {name: email} to CONTACT_PATH (JSON dict)"""
        data = {}
        if os.path.exists(CONTACT_PATH):
            try:
                with open(CONTACT_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except:
                pass  # corrupt → start fresh

        data[name.lower()] = email  # lowercase key to avoid duplicates

        try:
            with open(CONTACT_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"Contact saved to: {CONTACT_PATH}")
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"Could not save contact:\n{e}")

    def wait_till_click(self) -> tuple[bool, QPoint]:
        loop = QEventLoop()
        clicked = [False]
        pos = [QPoint()]

        def on_click(point: QPoint):
            clicked[0] = True
            pos[0] = point
            loop.quit()

        self.clicked_signal.connect(on_click)
        loop.exec()  # ← This will be quit by closeEvent

        try:
            self.clicked_signal.disconnect(on_click)
        except:
            pass

        return clicked[0], pos[0]

    def closeEvent(self, event):
        self.isOrbOpen = False

        # Let the base class handle closing
        super().closeEvent(event)

    def _quit_gracefully(self):
        self.close()

class ExecutableInputDialog(QDialog):
    """Dialog: [Full path]  <-->  [Keyword]"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Executable")
        self.setModal(True)
        self.resize(760, 170)
        self.setStyleSheet(self._stylesheet())

        # ---------- Layout ----------
        main = QVBoxLayout(self)
        main.setSpacing(16)
        main.setContentsMargins(24, 24, 24, 20)

        # Title
        title = QLabel("Add a program you can launch")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #1a1a1a;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main.addWidget(title)

        # ---- Row: Path | Browse | Keyword ----
        row = QHBoxLayout()
        row.setSpacing(12)

        # ----- Full path -----
        row.addWidget(QLabel("Path:"), 0)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("C:/Program Files/Notepad++/notepad++.exe")
        self.path_edit.setMinimumHeight(38)
        row.addWidget(self.path_edit, 4)

        browse = QPushButton()
        browse.setIcon(QIcon.fromTheme("document-open"))
        browse.setToolTip("Browse for executable")
        browse.setFixedSize(38, 38)
        browse.clicked.connect(self._pick_file)
        row.addWidget(browse, 0)

        # ----- Keyword -----
        row.addWidget(QLabel("Keyword:"), 0)
        self.key_edit = QLineEdit()
        self.key_edit.setPlaceholderText("notepad")
        self.key_edit.setMinimumHeight(38)
        row.addWidget(self.key_edit, 2)

        # ----- Icon preview -----
        self.icon_lbl = QLabel()
        self.icon_lbl.setFixedSize(32, 32)
        self.icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_lbl.setStyleSheet("background:transparent;")
        row.addWidget(self.icon_lbl, 0)

        main.addLayout(row)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )

        # Get buttons
        ok_btn = btns.button(QDialogButtonBox.StandardButton.Ok)
        cancel_btn = btns.button(QDialogButtonBox.StandardButton.Cancel)

        # Make OK the default button
        ok_btn.setDefault(True)

        # === Dark style for OK button ===
        ok_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #555;
                padding: 5px 15px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
            }
            QPushButton:pressed {
                background-color: #1d1d1d;
            }
            QPushButton:disabled {
                background-color: #444;
                color: #888;
            }
        """)

        # === Dark style for Cancel button (slightly lighter to distinguish) ===
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #444;
                color: #ddd;
                border: 1px solid #666;
                padding: 5px 15px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QPushButton:pressed {
                background-color: #333;
            }
        """)

        # Connect signals
        btns.accepted.connect(self._validate_and_accept)
        btns.rejected.connect(self.reject)

        main.addWidget(btns)

        btns.accepted.connect(self._validate_and_accept)
        btns.rejected.connect(self.reject)
        main.addWidget(btns)

        # Live icon preview
        self.path_edit.textChanged.connect(self._update_icon)

    # ------------------------------------------------------------------
    def _pick_file(self):
        start = self.path_edit.text().strip() or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Executable", start,
            "Executables (*.exe *.bat *.cmd *.py *.sh);;All Files (*)"
        )
        if path:
            self.path_edit.setText(path.replace("\\", "/"))

    # ------------------------------------------------------------------
    def _update_icon(self):
        path = self.path_edit.text().strip()
        if not path:
            self.icon_lbl.clear()
            return

        icon = QIcon(path)
        if icon.isNull():
            icon = QIcon.fromTheme("application-x-executable")
        self.icon_lbl.setPixmap(icon.pixmap(24, 24))

    # ------------------------------------------------------------------
    def _validate_and_accept(self):
        path = self.path_edit.text().strip()
        key  = self.key_edit.text().strip().lower()

        if not path:
            QMessageBox.warning(self, "Missing", "Please enter the full path.")
            return
        if not key:
            QMessageBox.warning(self, "Missing", "Please enter a keyword.")
            return

        p = Path(path)
        if not p.is_file():
            QMessageBox.warning(self, "Not found", f"File does not exist:\n{path}")
            return

        # Windows: any file works; *nix: check execute bit
        if not sys.platform.startswith("win") and not os.access(p, os.X_OK):
            QMessageBox.warning(self, "Not executable",
                                f"The file exists but is not marked executable.\n{path}")
            return

        self._path = path
        self._keyword = key
        self.accept()

    # ------------------------------------------------------------------
    def get_data(self) -> tuple[str, str]:
        """Return (full_path, keyword)"""
        return self._path, self._keyword

    # ------------------------------------------------------------------
    @staticmethod
    def _stylesheet() -> str:
        return """
        QDialog {background:#ffffff; border:1px solid #ced4da; border-radius:12px;}
        QLabel {color:#1a1a1a; font-size:13px; font-weight:600;}
        QLineEdit {
            padding:10px 14px; border:2px solid #adb5bd; border-radius:10px;
            background:#ffffff; color:#1a1a1a; font-size:14px;
        }
        QLineEdit:focus {border-color:#0d6efd; background:#f8fbff;}
        QLineEdit::placeholder {color:#6c757d; font-style:italic;}
        QPushButton {
            background:#e9ecef; border:2px solid #ced4da; border-radius:10px;
        }
        QPushButton:hover {background:#dee2e6;}
        QPushButton:pressed {background:#ced4da;}
        QDialogButtonBox QPushButton {
            min-width:90px; min-height:38px; font-weight:600;
            border-radius:10px; padding:8px 18px; color:white;
        }
        QDialogButtonBox QPushButton#qt_dialogbuttonbox_ok {
            background:#0d6efd; border:none;
        }
        QDialogButtonBox QPushButton#qt_dialogbuttonbox_ok:hover {background:#0b5ed7;}
        QDialogButtonBox QPushButton#qt_dialogbuttonbox_cancel {
            background:#6c757d;
        }
        QDialogButtonBox QPushButton#qt_dialogbuttonbox_cancel:hover {background:#5a6268;}
        """

class ContactInputDialog(QDialog):
    """Dialog: [Name]  <-->  [Email]  → saved to CONTACT_PATH"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Contact")
        self.setModal(True)
        self.resize(600, 160)
        self.setStyleSheet(ExecutableInputDialog._stylesheet())   # ← REUSE style

        main = QVBoxLayout(self)
        main.setSpacing(16)
        main.setContentsMargins(24, 24, 24, 20)

        # Title
        title = QLabel("Add a contact")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #1a1a1a;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main.addWidget(title)

        # ---- Row: Name | Email ----
        row = QHBoxLayout()
        row.setSpacing(12)

        row.addWidget(QLabel("Name:"), 0)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("John Doe")
        self.name_edit.setMinimumHeight(38)
        row.addWidget(self.name_edit, 3)

        row.addWidget(QLabel("Email:"), 0)
        self.email_edit = QLineEdit()
        self.email_edit.setPlaceholderText("john@example.com")
        self.email_edit.setMinimumHeight(38)
        row.addWidget(self.email_edit, 3)

        main.addLayout(row)

        # Buttons
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        ok_btn = btns.button(QDialogButtonBox.StandardButton.Ok)
        cancel_btn = btns.button(QDialogButtonBox.StandardButton.Cancel)
        ok_btn.setDefault(True)

        # Dark OK / Cancel (same as ExecutableInputDialog)
        ok_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d2d; color: white; border: 1px solid #555;
                padding: 5px 15px; border-radius: 4px; font-weight: bold;
            }
            QPushButton:hover {background-color: #3d3d3d;}
            QPushButton:pressed {background-color: #1d1d1d;}
        """)
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #444; color: #ddd; border: 1px solid #666;
                padding: 5px 15px; border-radius: 4px;
            }
            QPushButton:hover {background-color: #555;}
            QPushButton:pressed {background-color: #333;}
        """)

        btns.accepted.connect(self._validate_and_accept)
        btns.rejected.connect(self.reject)
        main.addWidget(btns)

    # ------------------------------------------------------------------
    def _validate_and_accept(self):
        name = self.name_edit.text().strip()
        email = self.email_edit.text().strip().lower()

        if not name:
            QMessageBox.warning(self, "Missing", "Please enter a name.")
            return
        if not email or "@" not in email:
            QMessageBox.warning(self, "Invalid", "Please enter a valid email.")
            return

        self._name = name
        self._email = email
        self.accept()

    # ------------------------------------------------------------------
    def get_data(self) -> tuple[str, str]:
        return self._name, self._email

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    orb = OrbWindow()
    orb.show()
    
    # --- Example: How to change the state ---
    states = ["idle", "listening", "talking", "processing"]
    current_state_index = 0
    
    def change_state():
        global current_state_index
        current_state_index = (current_state_index + 1) % len(states)
        state = states[current_state_index]
        print(f"Changing state to: {state}")
        orb.set_state(state)

    # Set up a timer to call change_state every 3 seconds (3000 ms)
    state_timer = QTimer()
    state_timer.timeout.connect(change_state)
    state_timer.start(3000)

    sys.exit(app.exec())