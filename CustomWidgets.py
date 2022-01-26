import sys
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtCore, QtGui, QtWidgets
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


class CollapsibleBox(QtWidgets.QWidget):
    def __init__(self, title="", parent=None):
        super(CollapsibleBox, self).__init__(parent)

        self.toggle_button = QtWidgets.QToolButton(
            text=title, checkable=True, checked=False
        )
        #self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonTextBesideIcon
        )
        self.toggle_button.setArrowType(QtCore.Qt.RightArrow)
        self.toggle_button.pressed.connect(self.on_pressed)

        self.toggle_animation = QtCore.QParallelAnimationGroup(self)

        self.content_area = QtWidgets.QScrollArea(
            maximumHeight=0, minimumHeight=0
        )
        self.content_area.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        self.content_area.setFrameShape(QtWidgets.QFrame.NoFrame)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toggle_button)
        lay.addWidget(self.content_area)

        self.toggle_animation.addAnimation(
            QtCore.QPropertyAnimation(self, b"minimumHeight")
        )
        self.toggle_animation.addAnimation(
            QtCore.QPropertyAnimation(self, b"maximumHeight")
        )
        self.toggle_animation.addAnimation(
            QtCore.QPropertyAnimation(self.content_area, b"maximumHeight")
        )

    @QtCore.pyqtSlot()
    def on_pressed(self):
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(
            QtCore.Qt.DownArrow if not checked else QtCore.Qt.RightArrow
        )
        self.toggle_animation.setDirection(
            QtCore.QAbstractAnimation.Forward
            if not checked
            else QtCore.QAbstractAnimation.Backward
        )
        self.toggle_animation.start()

    def setContentLayout(self, layout):
        lay = self.content_area.layout()
        del lay
        self.content_area.setLayout(layout)
        collapsed_height = (
            self.sizeHint().height() - self.content_area.maximumHeight()
        )
        content_height = layout.sizeHint().height()
        for i in range(self.toggle_animation.animationCount()):
            animation = self.toggle_animation.animationAt(i)
            animation.setDuration(500)
            animation.setStartValue(collapsed_height)
            animation.setEndValue(collapsed_height + content_height)

        content_animation = self.toggle_animation.animationAt(
            self.toggle_animation.animationCount() - 1
        )
        content_animation.setDuration(500)
        content_animation.setStartValue(0)
        content_animation.setEndValue(content_height)