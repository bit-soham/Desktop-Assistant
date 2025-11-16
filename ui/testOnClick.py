# any_program.py
from orb_window import OrbWindow
from PyQt6.QtWidgets import QApplication
import sys

app = QApplication(sys.argv)
orb = OrbWindow()
orb.show()

while orb.isOrbOpen:
    clicked, pos = orb.wait_till_click()
    print("eer")

    if not orb.isOrbOpen:
        break


print("Done. Orb stays alive.")
sys.exit(app.exec())