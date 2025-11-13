# orb_controller.py
# Thread-safe controller for the orb UI to be used from main.py

from PyQt6.QtCore import QObject, pyqtSignal, QTimer, QMetaObject, Qt
from PyQt6.QtWidgets import QApplication
from ui.orb_window import OrbWindow
import sys
import threading

class OrbController(QObject):
    """
    Controller class that manages the orb UI and provides thread-safe signals
    for state changes from the main application.
    """
    # Signals for state changes
    state_changed = pyqtSignal(str)  # "idle", "listening", "talking", "processing"
    _start_requested = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.orb = None
        self.app = None
        self.ui_thread = None
        self._initialized = False
        
    def start_ui(self):
        """Start the UI in a separate thread."""
        def run_qt_app():
            # Create QApplication in this thread
            try:
                self.app = QApplication(sys.argv)
            except RuntimeError:
                # QApplication already exists
                self.app = QApplication.instance()
            
            # Create the orb window
            self.orb = OrbWindow()
            
            # Connect the signal to the orb's set_state method
            # Use QMetaObject.invokeMethod for thread-safe slot invocation
            self.state_changed.connect(self.orb.set_state, Qt.ConnectionType.QueuedConnection)
            
            # Show the orb
            self.orb.show()
            
            self._initialized = True
            print("DEBUG: Orb window created and shown")
            
            # Run the Qt event loop
            self.app.exec()
        
        # Start Qt in a daemon thread
        self.ui_thread = threading.Thread(target=run_qt_app, daemon=True)
        self.ui_thread.start()
        
    def set_state(self, state: str):
        """
        Thread-safe method to change the orb's state.
        Valid states: "idle", "listening", "talking", "processing"
        """
        # Wait for initialization
        max_wait = 50  # 5 seconds max
        wait_count = 0
        while not self._initialized and wait_count < max_wait:
            import time
            time.sleep(0.1)
            wait_count += 1
            
        # Emit signal which will be handled in the Qt thread
        if self.orb and self._initialized:
            self.state_changed.emit(state)
            
    def stop_ui(self):
        """Stop the UI."""
        if self.app:
            self.app.quit()
