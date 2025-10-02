# status_bar.py

from PyQt6.QtWidgets import QStatusBar

def create_status_bar(parent):
    status_bar = QStatusBar(parent)
    status_bar.showMessage("Ready")
    return status_bar
