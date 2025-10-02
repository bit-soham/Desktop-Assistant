from PyQt6.QtWidgets import QMenuBar, QMenu
from PyQt6.QtGui import QAction
def create_menu_bar(parent):
    menu_bar = QMenuBar(parent)

    # File menu
    file_menu = QMenu("File", parent)
    new_action = QAction("New", parent)
    open_action = QAction("Open", parent)
    save_action = QAction("Save", parent)
    exit_action = QAction("Exit", parent)
    exit_action.triggered.connect(parent.close)

    file_menu.addAction(new_action)
    file_menu.addAction(open_action)
    file_menu.addAction(save_action)
    file_menu.addSeparator()
    file_menu.addAction(exit_action)
    menu_bar.addMenu(file_menu)

    # Edit menu
    edit_menu = QMenu("Edit", parent)
    undo_action = QAction("Undo", parent)
    redo_action = QAction("Redo", parent)
    cut_action = QAction("Cut", parent)
    copy_action = QAction("Copy", parent)
    paste_action = QAction("Paste", parent)

    edit_menu.addAction(undo_action)
    edit_menu.addAction(redo_action)
    edit_menu.addSeparator()
    edit_menu.addAction(cut_action)
    edit_menu.addAction(copy_action)
    edit_menu.addAction(paste_action)
    menu_bar.addMenu(edit_menu)

    # Help menu
    help_menu = QMenu("Help", parent)
    about_action = QAction("About", parent)

    help_menu.addAction(about_action)
    menu_bar.addMenu(help_menu)

    return menu_bar
