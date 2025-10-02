from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtGui import QAction

app = QApplication([])
window = QMainWindow()

action = QAction("Test Action", window)
menu = window.menuBar().addMenu("Test Menu")
menu.addAction(action)

window.show()
app.exec()
