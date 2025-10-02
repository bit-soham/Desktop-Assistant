import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from components.menu_bar import create_menu_bar
from components.status_bar import create_status_bar

class MainUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Desktop Assistant")
        self.setGeometry(100, 100, 800, 600)

        # Set up central widget
        central_widget = QWidget()
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Add components
        self.setMenuBar(create_menu_bar(self))
        self.setStatusBar(create_status_bar(self))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainUI()
    main_window.show()
    sys.exit(app.exec())
