# orb_window.py
# A fully dynamic and animated UI for "Athena"
#
# Contains fixes for:
# 1. All previous TypeError and NameError issues.
# 2. ## UPGRADE: Decouples "talking" animation logic from render speed
#    for a smoother, less chaotic effect.

import sys, math, random
from PyQt6.QtWidgets import QApplication, QWidget, QMenu
from PyQt6.QtGui import (
    QPainter, QColor, QAction, QRadialGradient, QPen, QBrush, QMouseEvent, QRegion
)
from PyQt6.QtCore import Qt, QTimer, QPoint, QPointF, QRectF

class OrbWindow(QWidget):
    
    def __init__(self):
        super().__init__()

        # --- 1. Window Setup ---
        self.orb_size = 150  # 150px
        self.setFixedSize(self.orb_size, self.orb_size)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setToolTip("Athena")

        # --- 2. State and Animation Setup ---
        self.current_state = "idle"
        self.animation_frame = 0  # A counter that drives all animations

        # This timer will trigger a repaint (a new frame) ~60 times/sec
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(16) # 1000ms / 60fps = ~16ms

        # ## UPGRADE: Add class attributes to store talking bar heights
        self.talk_h1 = 10.0
        self.talk_h2 = 10.0
        self.talk_h3 = 10.0

        # ## UPGRADE: Create a second, slower timer just for the talking logic
        self.talking_timer = QTimer(self)
        self.talking_timer.timeout.connect(self.update_talking_animation)
        self.talking_timer.setInterval(100) # Update 10 times/sec (every 100ms)

        # We must set a circular mask for the window
        self._set_circular_mask()

        # --- 3. Mouse Dragging ---
        self.drag_pos = None

    def _set_circular_mask(self):
        """Sets a simple, clean circular mask for the window using QRegion."""
        region = QRegion(self.rect(), QRegion.RegionType.Ellipse)
        self.setMask(region)

    def set_state(self, state: str):
        """
        Public method to change the orb's animation state.
        Valid states: "idle", "listening", "talking", "processing"
        """
        if state not in ["idle", "listening", "talking", "processing"]:
            state = "idle"

        self.current_state = state

        # ## UPGRADE: Start or stop the talking_timer based on the new state
        if self.current_state == "talking":
            self.talking_timer.start()
        else:
            self.talking_timer.stop()

    def update_animation(self):
        """The main animation loop (60 FPS)."""
        self.animation_frame += 1
        self.update() # This triggers the paintEvent

    # ## UPGRADE: New function to calculate talking state at 10 FPS
    def update_talking_animation(self):
        """
        This function is called by the talking_timer 10 times/sec.
        It calculates the new bar heights.
        """
        # In a real app, you would get this from an audio input stream
        self.talk_h1 = 10 + (random.randint(0, 100) / 100) * 45
        self.talk_h2 = 10 + (random.randint(0, 100) / 100) * 70
        self.talk_h3 = 10 + (random.randint(0, 100) / 100) * 45

    def paintEvent(self, event):
        """
        This is the core function where all drawing happens (60 FPS).
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Clear the widget area with full transparency
        painter.fillRect(self.rect(), Qt.GlobalColor.transparent)

        # Call the correct drawing function based on the current state
        if self.current_state == "idle":
            self.draw_idle(painter)
        elif self.current_state == "listening":
            self.draw_listening(painter)
        elif self.current_state == "talking":
            self.draw_talking(painter)
        elif self.current_state == "processing":
            self.draw_processing(painter)

    # --- 4. State-Specific Drawing Functions ---

    def draw_idle(self, painter: QPainter):
        """Draws a gently "breathing" orb."""
        pulse = (math.sin(self.animation_frame * 0.05) + 1) / 2 # Normalized (0 to 1)
        glow_size = self.orb_size * 0.7 + (pulse * 10)
        
        gradient = QRadialGradient(QPointF(self.rect().center()), self.orb_size / 2)
        gradient.setColorAt(0, QColor(0, 80, 150))
        gradient.setColorAt(1, QColor(0, 20, 50))
        painter.setBrush(gradient)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(self.rect())
        
        glow_color = QColor(0, 150, 255, int(30 + (pulse * 30))) 
        painter.setBrush(QBrush(glow_color))
        center = self.rect().center()
        painter.drawEllipse(center, int(glow_size / 2), int(glow_size / 2))

    def draw_listening(self, painter: QPainter):
        """Draws a brighter, more "attentive" orb."""
        pulse = (math.sin(self.animation_frame * 0.15) + 1) / 2 # Faster pulse
        core_size = self.orb_size * 0.5 + (pulse * 10)
        
        gradient = QRadialGradient(QPointF(self.rect().center()), self.orb_size / 2)
        gradient.setColorAt(0, QColor(0, 150, 255))
        gradient.setColorAt(1, QColor(0, 50, 120))
        painter.setBrush(gradient)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(self.rect())

        core_color = QColor(200, 230, 255, int(150 + (pulse * 50)))
        painter.setBrush(QBrush(core_color))
        center = self.rect().center()
        painter.drawEllipse(center, int(core_size / 2), int(core_size / 2))

    def draw_talking(self, painter: QPainter):
        """
        Draws a simple 3-bar spectrum visualizer.
        ## UPGRADE: This function now ONLY draws. It no longer calculates.
        """
        gradient = QRadialGradient(QPointF(self.rect().center()), self.orb_size / 2)
        gradient.setColorAt(0, QColor(0, 120, 220))
        gradient.setColorAt(1, QColor(0, 40, 100))
        painter.setBrush(gradient)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(self.rect())

        bar_width = 15
        center_x = self.rect().center().x()
        center_y = self.rect().center().y()
        
        # ## UPGRADE: Read from the class attributes (self.talk_h1)
        #    instead of calculating new random numbers here.
        h1 = self.talk_h1
        h2 = self.talk_h2
        h3 = self.talk_h3

        painter.setBrush(QColor(200, 230, 255))

        bar1 = QRectF(center_x - bar_width * 1.5, center_y - h1 / 2, bar_width, h1)
        bar2 = QRectF(center_x - bar_width * 0.5, center_y - h2 / 2, bar_width, h2)
        bar3 = QRectF(center_x + bar_width * 0.5, center_y - h3 / 2, bar_width, h3)

        painter.drawRect(bar1)
        painter.drawRect(bar2)
        painter.drawRect(bar3)

    def draw_processing(self, painter: QPainter):
        """Draws a spinning "radar" arc. For tracking, notes, etc."""
        gradient = QRadialGradient(QPointF(self.rect().center()), self.orb_size / 2)
        gradient.setColorAt(0, QColor(130, 0, 180)) # Purple "thinking" color
        gradient.setColorAt(1, QColor(40, 0, 60))
        painter.setBrush(gradient)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(self.rect())

        angle = (self.animation_frame * 3) % 360
        pen = QPen(QColor(230, 200, 255), 10)
        painter.setPen(pen)
        
        painter.drawArc(self.rect().adjusted(15, 15, -15, -15), angle * 16, 90 * 16)

    # --- 5. Mouse and Menu Events ---

    def mousePressEvent(self, event: QMouseEvent):
        """Captures the mouse-click position."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_pos = event.globalPosition().toPoint() - self.pos()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        """Moves the window if the mouse is being dragged."""
        if event.buttons() == Qt.MouseButton.LeftButton and self.drag_pos:
            new_pos = event.globalPosition().toPoint() - self.drag_pos
            self.move(new_pos)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Resets the drag position."""
        self.drag_pos = None
        event.accept()

    def contextMenuEvent(self, event: QMouseEvent):
        """Right-click menu for 'Athena' and 'Quit'."""
        context_menu = QMenu(self)
        
        title_action = QAction("Athena", self)
        title_action.setEnabled(False)
        context_menu.addAction(title_action)
        
        context_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(QApplication.instance().quit)
        
        context_menu.addAction(quit_action)
        
        context_menu.exec(event.globalPos())


# --- This is the "main" part that runs the application ---
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