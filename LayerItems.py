#==================================================================
#==================================================================
from shutil import copy
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QBrush, QColor, QCursor, QImage, QKeyEvent, QKeySequence, QPixmap, QPainter, QPen
from PyQt5.QtWidgets import QGraphicsEllipseItem, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QShortcut
import numpy as np
import cv2
import math
import copy

from numpy.typing import _16Bit

from utils import pixmap_to_numpy
#==================================================================
#==================================================================

#-----------------------------------------------------------------
class LayerItem(QtWidgets.QGraphicsRectItem):
    DrawState, EraseState, FillState, MagneticLasso = range(4)

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

        #Grab-cut
        self.background_gc_layer = None;
        self.foreground_gc_layer = None;
        self.mask_gc = None;

        #Magnetic lasso
        self.__magnetic_lasso_active = False;
        self.__magnetic_lasso_image_state = None;
        #image before we start drawing anything,
        #we use it for cancelation
        self.__magnetic_lasso_original_image = None;
        self.__magnetic_lasso_tool = None;
        self.__magnetic_lasso_undo_list = [];
        self.__magnetic_lasso_prev_point = None;

    @property
    def name(self):
        return self.m_name;
    
    @name.setter
    def name(self,n):
        self.m_name = n;

    def reset(self):
        r = self.parentItem().pixmap().rect();
        self.setRect(QtCore.QRectF(r))
        self.m_pixmap = QtGui.QPixmap(r.size())
        self.m_pixmap.fill(QtCore.Qt.transparent)
        self.reset_mask_gc();
    
    
    def clear(self):
        self.m_pixmap.fill(QtCore.Qt.transparent)

    def reset_mask_gc(self):
        self.mask_gc = np.zeros(shape=(self.m_pixmap.height(), self.m_pixmap.width()), dtype=np.uint8);
        self.mask_gc[:,:] = cv2.GC_PR_BGD;
    
    def reset_lasso_tool(self, pixmap):

        img = pixmap_to_numpy(pixmap);
        self.__current_image_width = img.shape[1];
        self.__current_image_height = img.shape[0];
        
        self.__magnetic_lasso_tool = cv2.segmentation_IntelligentScissorsMB();
        self.__magnetic_lasso_tool.setEdgeFeatureCannyParameters(32, 100);
        self.__magnetic_lasso_tool.setGradientMagnitudeMaxLimit(200);
        img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)));
        self.__magnetic_lasso_tool.applyImage(img);
        self.__magnetic_lasso_undo_list.clear();

    def paint(self, painter, option, widget=None):
        if self.m_visible:
            #painter.save()
            painter.drawPixmap(QtCore.QPoint(), self.m_pixmap)
            #painter.restore()
        super(LayerItem, self).paint(painter, option, widget)

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
            elif self.m_current_state == LayerItem.FillState:
                self._fill_region(event.pos(), QtGui.QPen(self.pen_color, 1 ,QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin));
            elif self.m_current_state == LayerItem.MagneticLasso:
                scene_pos = self.mapToScene(event.pos());
                if scene_pos.x() > 0 \
                and scene_pos.y() > 0 \
                and scene_pos.x() < self.__current_image_width \
                and scene_pos.y() < self.__current_image_height:

                    if self.__magnetic_lasso_active is False\
                    and scene_pos.x() > 0 \
                    and scene_pos.y() > 0 \
                    and scene_pos.x() < self.__current_image_width \
                    and scene_pos.y() < self.__current_image_height:
                        self.__magnetic_lasso_tool.buildMap((int(scene_pos.x()/2), int(scene_pos.y()/2)));
                        self.__magnetic_lasso_active = True;
                        self.__magnetic_lasso_image_state = self.m_pixmap.copy(QtCore.QRect());
                        self.__magnetic_lasso_original_image = self.m_pixmap.copy(QtCore.QRect());
                        self.__magnetic_lasso_undo_list.append([self.__magnetic_lasso_image_state.copy(QtCore.QRect()), ((int(scene_pos.x()), int(scene_pos.y())))]);
                        self.__magnetic_lasso_prev_point = (int(scene_pos.x()), int(scene_pos.y()));
                    else:
                        self.__magnetic_lasso_tool.buildMap((int(scene_pos.x()/2), int(scene_pos.y()/2)));
                        self.__magnetic_lasso_undo_list.append([self.__magnetic_lasso_image_state.copy(QtCore.QRect()), self.__magnetic_lasso_prev_point]);
                        self.__magnetic_lasso_image_state = self.m_pixmap.copy(QtCore.QRect());
                        self.__magnetic_lasso_prev_point = (int(scene_pos.x()), int(scene_pos.y()));

        elif self.m_mid_mouse_down and self.m_active and self.m_visible:
            if self.m_current_state == LayerItem.MagneticLasso:
                if len(self.__magnetic_lasso_undo_list) != 0:
                    prev_magnetic_lasso_state = self.__magnetic_lasso_undo_list.pop();
                    self.__magnetic_lasso_tool.buildMap((prev_magnetic_lasso_state[1][0], prev_magnetic_lasso_state[1][1]));
                    self.__magnetic_lasso_image_state = prev_magnetic_lasso_state[0];
                    self.m_pixmap = prev_magnetic_lasso_state[0];
                    self.update();
        
        super(LayerItem, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        #print(self);
        if self.m_left_mouse_down and self.m_active and self.m_visible:
            if self.m_current_state == LayerItem.EraseState:
                self._clear(self.mapToScene(event.pos()), QtGui.QPen(self.pen_color, self.pen_thickness,QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
            elif self.m_current_state == LayerItem.DrawState:
                self.m_line_draw.setP2(event.pos())
                self._draw_line(
                    self.m_line_draw, QtGui.QPen(self.pen_color, self.pen_thickness,QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
                )
                self.m_line_draw.setP1(event.pos());

        #Magnetic lasso
        if self.m_current_state == LayerItem.MagneticLasso:
            scene_pos = self.mapToScene(event.pos());
            if self.__magnetic_lasso_active is True \
            and scene_pos.x() > 0 \
            and scene_pos.y() > 0 \
            and scene_pos.x() < self.__current_image_width \
            and scene_pos.y() < self.__current_image_height:
                image = self.__magnetic_lasso_image_state.toImage();
                path = self.__magnetic_lasso_tool.getContour((int(scene_pos.x()/2),int(scene_pos.y()/2)));
                path = path.squeeze();
                for idx in range(len(path)-1):
                    image.setPixelColor(int(path[idx][0]*2),int(path[idx][1]*2), self.pen_color);
                    image.setPixelColor(int((path[idx+1][0]*2)*0.5 + (path[idx][0]*2)*0.5),int((path[idx][1]*2)*0.5 + (path[idx+1][1]*2)*0.5), self.pen_color);
                
                image.setPixelColor(int(path[len(path)-1][0]*2),int(path[len(path)-1][1]*2), self.pen_color);
                px = QPixmap();
                px.convertFromImage(image);
                self.m_pixmap = px;
                self.update();
        #-------------------------------------------------------------
        super(LayerItem, self).mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        #Update mouse buttons state
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.m_left_mouse_down = False;
            #Since we handle undo of magnetic lasso during drawing 
            #differently, only add to the stack when we are not
            #in magnetic lasso state
            if self.m_current_state != LayerItem.MagneticLasso:
                self.undostack.append(self.pix_before);

        elif event.button() == QtCore.Qt.MouseButton.MidButton:
            self.m_mid_mouse_down = False;
        #--------------------------------------------------

        super(LayerItem, self).mouseReleaseEvent(event)
    
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.cancel_magnetic_lasso();
        elif event.key() == QtCore.Qt.Key.Key_Enter or event.key() == QtCore.Qt.Key.Key_Return:
            self.accept_magnetic_lasso();
        return super().keyPressEvent(event)
    
    def undo_event(self):
        #Since we are handling undo of magnetic lasso differently
        #do not undo when we are working with this tool
        if self.m_active and self.m_visible and self.__magnetic_lasso_active is False:
            if(len(self.undostack) != 0):
                self.redostack.append(self.m_pixmap.copy(self.m_pixmap.rect()));
                self.m_pixmap = self.undostack.pop();
                self.update();
    
    def cancel_magnetic_lasso(self):
        self.__magnetic_lasso_prev_point = None;
        self.__magnetic_lasso_undo_list.clear();
        self.__magnetic_lasso_active = False;
        self.m_pixmap = self.__magnetic_lasso_original_image.copy(QtCore.QRect());
        self.update();

    def accept_magnetic_lasso(self):
        self.__magnetic_lasso_prev_point = None;
        self.__magnetic_lasso_undo_list.clear();
        self.__magnetic_lasso_active = False;
        self.undostack.append(self.__magnetic_lasso_original_image.copy(QtCore.QRect()));
        self.m_pixmap = self.__magnetic_lasso_image_state.copy(QtCore.QRect());
        self.update();
    
    def redo_event(self):
        #Since we are handling undo of magnetic lasso differently
        #do not undo when we are working with this tool
        if self.m_active and self.m_visible and self.__magnetic_lasso_active is False:
            if(len(self.redostack) != 0):
                self.undostack.append(self.m_pixmap.copy(self.m_pixmap.rect()));
                self.m_pixmap = self.redostack.pop();
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

    def _get_pixel(self, x , y, pixels, w):
        i = (x + (y * w)) * 4
        return pixels[i:i + 3]
    
    def _get_cardinal_points(self, have_seen, center_pos, w, h, pixels, rgb):
        s = pixels[pixels == np.array(rgb)];
        ss = s.sum();
        points = []
        cx, cy = center_pos
        pix = pixels[cy, cx];
        for x, y in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            xx, yy = cx + x, cy + y
            if (xx >= 0 and xx < w and
                yy >= 0 and yy < h and
                (xx, yy) not in have_seen):
                if(rgb[0] != pix[0] or rgb[1] != pix[1] or rgb[2] != pix[2]):
                    points.append((xx, yy))
                    have_seen.add((xx, yy))

        return points

    def _fill_region(self, pos, pen):
        dividing_factor = 2;
        pixels = pixmap_to_numpy(self.m_pixmap);
        w,h = pixels.shape[1], pixels.shape[0];
        pixels = cv2.resize(pixels, (int(h/dividing_factor), int(w/dividing_factor)));

        pos = QtCore.QPoint(int(pos.x()/dividing_factor), int(pos.y()/dividing_factor));
        queue = [(int(pos.x()/dividing_factor),int(pos.y()/dividing_factor))];

        rgb = self.pen_color.getRgb();
        r,g,b = (rgb[2]), (rgb[1]), (rgb[0]);

        seen = set();
        painter = QtGui.QPainter(self.m_pixmap);
        painter.setPen(pen);

        while(len(queue) != 0):
            #get and fill the current point
            x,y = queue.pop();
            painter.drawPoint(QtCore.QPoint(x*dividing_factor, y*dividing_factor));

            a = self._get_cardinal_points(seen, (int(x/dividing_factor),int(y/dividing_factor)), int(w/dividing_factor), int(h/dividing_factor), pixels, [r,g,b]);
            queue.extend(a);

        painter.end();
        self.update();
    
    def _check_condition(self, p, w, h, pixels, rgb):
        x,y = p;
        pix = self._get_pixel(x,y,pixels,w);

        if(x > 0 and x < w and y > 0 and y < h):
            if(rgb[0] == pix[0] and rgb[1] == pix[1] and rgb[2] == pix[2]):
                return False;
            return True;
        return False;

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
        self.setActive(b);
        self.setEnabled(b);

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
    DrawState, EraseState, FillState, MagneticLasso = range(4)
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
            elif self.m_current_state == self.EraseState:
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