import sys
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication, QGridLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow, QPushButton, QVBoxLayout, QWidget

class LabelledRadListItem (QWidget):
    open_radiograph_signal = pyqtSignal(str);
    delete_radiograph_signal = pyqtSignal(str);
    def __init__ (self, parent = None):
        super(LabelledRadListItem, self).__init__(parent);
        self.grid_layout = QGridLayout();
        self.name    = QLabel()
        self.status  = QLabel()
        self.grid_layout.addWidget(self.name,0,0,2,2);
        self.grid_layout.addWidget(self.status,1,0,2,2)
        self.open_button = QPushButton('Open');
        self.open_button.clicked.connect(self.open_radiograph_slot);
        self.delete_button = QPushButton('Delete');
        self.delete_button.clicked.connect(self.delete_radiograph_slot);
        self.grid_layout.addWidget(self.open_button,0,2,1,1)
        self.open_button.setFixedWidth(5)
        self.grid_layout.addWidget(self.delete_button,0,3,1,1);
        self.grid_layout.setHorizontalSpacing(3);
        self.setLayout(self.grid_layout)
        

    def set_name (self, text, color):
        self.name.setStyleSheet(f'color: rgb{color};')
        self.name.setText(text)

    def set_status (self, text, color):
        self.status.setStyleSheet(f'color: rgb{color};')
        self.status.setText(text)

    def open_radiograph_slot(self):
        self.open_radiograph_signal.emit(self.name.text());
    
    def delete_radiograph_slot(self):
        self.delete_radiograph_signal.emit(self.name.text());