#==================================================================
#==================================================================
from shutil import copy
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QBrush, QColor, QCursor, QImage, QPixmap, QPainter, QPen
from PyQt5.QtWidgets import QGraphicsEllipseItem, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem
import numpy as np
import cv2
import math
import copy
#==================================================================
#==================================================================

#-----------------------------------------------------------------
class LayerItem(QtWidgets.QGraphicsRectItem):
    DrawState, EraseState = range(2)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPen(QtGui.QPen(QtCore.Qt.NoPen))

        self.m_line_eraser = QtCore.QLineF()
        self.m_line_draw = QtCore.QLineF()
        self.m_pixmap = QtGui.QPixmap()

        self.m_left_mouse_down = False;
        self.m_right_mouse_down = False;
        self.m_mid_mouse_down = False;
        self.m_active = False;
        self.m_visible = True;
        self.m_current_state = LayerItem.DrawState
        self._pen_color = QColor(255,255,255,255);


        self.m_name = "";

        self.undostack = [];
        self.redostack = [];
        self.pix_before = None;

    @property
    def name(self):
        return self.m_name;
    
    @name.setter
    def name(self,n):
        self.m_name = n;

    def reset(self):
        r = self.parentItem().pixmap().rect()
        self.setRect(QtCore.QRectF(r))
        self.m_pixmap = QtGui.QPixmap(r.size())
        self.m_pixmap.fill(QtCore.Qt.transparent)
        #self.orig = 

    def paint(self, painter, option, widget=None):
        if self.m_visible:
            #painter.save()
            painter.drawPixmap(QtCore.QPoint(), self.m_pixmap)
            #painter.restore()
        super().paint(painter, option, widget)

    def mousePressEvent(self, event):
        #Update mouse buttons state
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            self.m_right_mouse_down = True;
        elif event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.m_left_mouse_down = True;
            self.pix_before = self.m_pixmap.copy(self.m_pixmap.rect());
            self.redostack.clear();
        elif event.button() == QtCore.Qt.MouseButton.MidButton:
            self.m_mid_mouse_down = True;
        #--------------------------------------------------

        if self.m_left_mouse_down and self.m_active and self.m_visible:
            if self.m_current_state == LayerItem.EraseState:
                self._clear(self.mapToScene(event.pos()), QtGui.QPen(self.pen_color, self.pen_thickness,QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
            elif self.m_current_state == LayerItem.DrawState:
                self.m_line_draw.setP1(event.pos())
                self.m_line_draw.setP2(event.pos())

        super().mousePressEvent(event)
        event.accept()
        
    def mouseMoveEvent(self, event):
        if self.m_left_mouse_down and self.m_active and self.m_visible:
            if self.m_current_state == LayerItem.EraseState:
                self._clear(self.mapToScene(event.pos()), QtGui.QPen(self.pen_color, self.pen_thickness,QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
            elif self.m_current_state == LayerItem.DrawState:
                self.m_line_draw.setP2(event.pos())
                self._draw_line(
                    self.m_line_draw, QtGui.QPen(self.pen_color, self.pen_thickness,QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
                )
                self.m_line_draw.setP1(event.pos())
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        #Update mouse buttons state
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.m_left_mouse_down = False;
            self.undostack.append(self.pix_before);

        elif event.button() == QtCore.Qt.MouseButton.MidButton:
            self.m_mid_mouse_down = False;
        #--------------------------------------------------

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if(self.m_active and self.m_visible):
            if event.key() == (QtCore.Qt.Key_Control and QtCore.Qt.Key_Y):
                if(len(self.redostack) != 0):
                    self.undostack.append(self.m_pixmap.copy(self.m_pixmap.rect()));
                    self.m_pixmap = self.redostack.pop();
                    self.update();
                pass
            if event.key() == (QtCore.Qt.Key_Control and QtCore.Qt.Key_Z):
                if(len(self.undostack) != 0):
                    self.redostack.append(self.m_pixmap.copy(self.m_pixmap.rect()));
                    self.m_pixmap = self.undostack.pop();
                    self.update();
            
    def _draw_line(self, line, pen):
        painter = QtGui.QPainter(self.m_pixmap)
        painter.setPen(pen)
        painter.drawLine(line)
        #print(line);
        painter.end()
        self.update()

    def _clear(self, pos, pen):
        painter = QtGui.QPainter(self.m_pixmap)
        painter.setPen(pen);
        r = QtCore.QRect(QtCore.QPoint(), self.pen_thickness * QtCore.QSize())
        r.moveCenter(QtCore.QPoint(pos.x(), pos.y()))
        #painter.save()
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
        painter.eraseRect(r)
        #painter.restore()
        painter.end()
        self.update()

    def get_numpy(self):
        image = self.m_pixmap.toImage()
        s = image.bits().asstring(image.width() * image.height() * 4)
        arr = np.fromstring(s, dtype=np.uint8).reshape((image.height(), image.width(), 4)) 
        return arr;

    @property
    def pen_thickness(self):
        return self._pen_thickness

    @pen_thickness.setter
    def pen_thickness(self, thickness):
        self._pen_thickness = thickness

    @property
    def active(self):
        return self.m_active;
    
    @active.setter
    def active(self, b):
        self.m_active = b;

    @property
    def visible(self):
        return self.m_visible;
    
    @visible.setter
    def visible(self, b):
        self.m_visible = b;

    @property
    def pen_color(self):
        return self._pen_color

    @pen_color.setter
    def pen_color(self, color):
        self._pen_color = color

    @property
    def current_state(self):
        return self.m_current_state

    @current_state.setter
    def current_state(self, state):
        self.m_current_state = state

#------------------------------------------------------------------

#------------------------------------------------------------------
class MouseLayer(QtWidgets.QGraphicsRectItem):
    DrawState, EraseState = range(2)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPen(QtGui.QPen(QtCore.Qt.NoPen))

        self.m_line_eraser = QtCore.QLineF()
        self.m_line_draw = QtCore.QLineF()
        self.m_pixmap = QtGui.QPixmap()

        self.m_left_mouse_down = False;
        self.m_right_mouse_down = False;
        self.m_mid_mouse_down = False;
        self.m_mouse_last_x = 0;
        self.m_mouse_last_y = 0;
        self.m_empty=True;
        self.m_pen_thickness = 150;

        self._pen_color = QColor(255,255,255,255);
        self.m_current_state = self.DrawState;

    def reset(self):
        r = self.parentItem().pixmap().rect()
        self.setRect(QtCore.QRectF(r))
        self.m_pixmap = QtGui.QPixmap(r.size())
        self.m_pixmap.fill(QtCore.Qt.transparent)

    def paint(self, painter, option, widget=None):
        #painter.save();
        painter.drawPixmap(QtCore.QPoint(), self.m_pixmap);
        #painter.restore();
        #painter.end();
        super().paint(painter, option, widget);

    def mousePressEvent(self, event):
        #Update mouse buttons state
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            self.m_right_mouse_down = True;
        elif event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.m_left_mouse_down = True;
        elif event.button() == QtCore.Qt.MouseButton.MidButton:
            self.m_mid_mouse_down = True;
        #--------------------------------------------------

        if self.m_left_mouse_down:
            self.m_line_draw.setP1(event.pos())
            self.m_line_draw.setP2(event.pos())

        super().mousePressEvent(event)
        event.accept()
        
    def update_cursor(self, pos):
        if not self.m_empty:
            self.m_pixmap.fill(QtCore.Qt.transparent)
            self.m_line_draw.setP2(pos)
            if self.m_current_state == self.DrawState:
                self._draw_line(
                    self.m_line_draw, QtGui.QPen(self.pen_color, self.pen_thickness,QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
                )
            else:
                self._draw_rect(pos);
            self.m_line_draw.setP1(pos)

    def _draw_line(self, line, pen):
        painter = QtGui.QPainter(self.m_pixmap)
        painter.setPen(pen)
        painter.drawLine(line)
        painter.end()
        self.update()
    
    def _draw_rect(self, pos):
        painter = QtGui.QPainter(self.m_pixmap)
        painter.fillRect(QtCore.QRectF(pos.x()-self.pen_thickness/2,pos.y()-self.pen_thickness/2, self.pen_thickness,self.pen_thickness), QtGui.QBrush(QColor(255,255,255)));
        painter.end()
        self.update()
    
    def keyPressEvent(self, event):
        super().keyPressEvent(event);

    @property
    def pen_thickness(self):
        return self.m_pen_thickness

    @pen_thickness.setter
    def pen_thickness(self, thickness):
        self.m_pen_thickness = thickness

    @property
    def pen_color(self):
        return self._pen_color

    @pen_color.setter
    def pen_color(self, color):
        self._pen_color = color
    
    @property
    def current_state(self):
        return self.m_current_state

    @current_state.setter
    def current_state(self, state):
        self.m_current_state = state
    
    @property
    def has_image(self):
        return not self.m_empty;
    
    @has_image.setter
    def has_image(self, b):
        self.m_empty = not b;

#------------------------------------------------------------------