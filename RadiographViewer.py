#==================================================================
#==================================================================
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QColor, QCursor, QImage, QPixmap, QPainter, QPen
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsPixmapItem
import numpy as np
from LayerItems import *
import math
import Config
#==================================================================
#==================================================================

class RadiographViewer(QtWidgets.QGraphicsView):
    size_changed_signal = QtCore.pyqtSignal(int);
    def __init__(self, parent):
        super().__init__(parent);
        
        self.m_zoom = 0;
        self.m_empty = True;
        self.setScene(QtWidgets.QGraphicsScene(self))

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse);
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse);
        self.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff);
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff);
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(0,0,0)));
        self.setFrameShape(QtWidgets.QFrame.NoFrame);

        self.background_item = QtWidgets.QGraphicsPixmapItem()

        self.m_mouse_layer = MouseLayer(self.background_item);
        color = QtGui.QColor(255,255,255);
        self.m_mouse_layer.pen_color = color
        self.m_mouse_layer.pen_thickness = 150;
        self.m_mouse_layer.setZValue(0)
        
        self.scene().addItem(self.background_item);
        
        self.m_left_mouse_down = False;
        self.m_right_mouse_down = False;
        self.m_mid_mouse_down = False;
        self.m_last_x_view = 0;
        self.m_last_y_view = 0;
        self.m_has_item = False;
        self.m_enable_drawing = True;
        self.m_brush_size = 150;
        self.m_mouse_layer_opacity = 1.0;

        self.layers = [];
        self.m_layer_count = 0;
        self.m_active_layer = -1;
        self.m_current_state = LayerItem.DrawState;

        self.coordinate_label = None;

    #Setter/Getter 
    @property
    def brush_size(self):
        return self.m_brush_size;
    
    @brush_size.setter
    def brush_size(self, size):
        self.m_brush_size = size;
        self.m_mouse_layer.pen_thickness = size;
        if self.m_active_layer != -1 and self.m_active_layer < self.m_layer_count:
            if self.m_active_layer != -1 and self.m_active_layer < self.m_layer_count:
                self.layers[self.m_active_layer].pen_thickness = size;
    @property
    def has_image(self):
        return not self.m_empty;
    
    @property
    def drawing_state(self):
        return self.m_enable_drawing;
    
    @property
    def layer_count(self):
        return self.m_layer_count;
        
    @drawing_state.setter
    def drawing_state(self, e):
        self.m_enable_drawing = e;
    
    def set_coordinate_label(self, lbl):
        self.coordinate_label = lbl;

    def reset_layer(self,idx):
        self.foreground_item.reset()
        pass

    def get_layer(self,idx):
        if idx < self.m_layer_count:
            return self.layers[idx];

    def get_active_layer(self):
        if self.m_active_layer < self.m_layer_count and self.m_active_layer != -1:
            return self.layers[self.m_active_layer];
    
    def get_layer_names(self):
        ret = [];
        for l in self.layers:
            ret.append(l.name);
        return ret;

    def set_state(self, state):
        self.m_current_state = state;
        if self.m_active_layer != -1 and self.m_active_layer < self.m_layer_count:
            self.layers[self.m_active_layer].current_state = state;
        
        #set state of the mouse layer
        self.m_mouse_layer.current_state = state;

    def set_layer_pixmap(self, idx, pixmap):
        self.layers[idx].m_pixmap = pixmap;
        pass

    def set_image(self, pixmap=None, reset = True):
        self.background_item.setPixmap(pixmap);
        if reset:
            self.scene().setSceneRect(
                QtCore.QRectF(QtCore.QPointF(), QtCore.QSizeF(pixmap.size()))
            )
            self.m_mouse_layer.reset();

            self.fitInView(self.background_item, QtCore.Qt.KeepAspectRatio)
            self.centerOn(self.background_item)

            self.m_empty= False;
            self.m_mouse_layer.has_image = True;
    
    '''
        We can hide a certain layer using this method.
        idx: index of the layer.
        v: visibility state
    '''
    def set_layer_visibility(self, idx, b):
        self.layers[idx].visible = b;
        self.update();
        self.background_item.update();
        self.layers[idx].update();

    '''
        Sets the opacity of the current active layer to the given value
    '''
    def set_layer_opacity(self, v):
        if self.m_active_layer != -1 and self.m_active_layer < self.m_layer_count:
            self.layers[self.m_active_layer].setOpacity(v/255);
    
    '''
        Sets the opacity of mouse layer
    '''
    def set_mouse_layer_opacity(self, v):
        self.m_mouse_layer_opacity = v/255;
        self.m_mouse_layer.setOpacity(self.m_mouse_layer_opacity );
        pass
    
    def set_active_layer(self, idx):
        #deactive other layers
        if self.m_active_layer < self.m_layer_count and self.m_active_layer != -1:
            self.layers[self.m_active_layer].ungrabMouse();
            self.layers[self.m_active_layer].ungrabKeyboard();

        for l in self.layers:
            l.active = False;
            l.setEnabled(False);
            l.setActive(False);
        #Active currently selected layer
        self.m_active_layer = idx;
        self.layers[idx].active = True;
        self.layers[idx].setEnabled(True);
        self.layers[idx].setActive(True);
        self.layers[idx].grabMouse();
        self.layers[idx].grabKeyboard();
        self.layers[idx].pen_thickness = self.m_brush_size;
        self.layers[idx].current_state = self.m_current_state;
        #ُSince color value may change, we should update mouse layer color
        self.m_mouse_layer.pen_color = self.layers[idx].pen_color;
    #-----------------------------------------------------

    def add_layer(self, name, color):
        foreground_item = LayerItem(self.background_item);
        foreground_item.pen_color = QColor(color[0],color[1],color[2]);
        foreground_item.pen_thickness = self.m_brush_size;
        foreground_item.current_state = self.m_current_state;
        foreground_item.setZValue(self.m_layer_count+1)
        foreground_item.reset();
        foreground_item.name = name;
        
        if self.m_active_layer < self.m_layer_count and self.m_active_layer != -1:
            self.layers[self.m_active_layer].ungrabMouse();
            self.layers[self.m_active_layer].ungrabKeyboard();

        #deactive other layers and active currently added layer
        for l in self.layers:
            l.active = False;
            l.setEnabled(False);
            l.setActive(False);
        foreground_item.active = True;
        foreground_item.setEnabled(True);
        foreground_item.setActive(True);
        foreground_item.grabMouse();
        foreground_item.grabKeyboard();
    
        self.layers.append(foreground_item);
        self.m_active_layer = len(self.layers)-1;
        self.m_layer_count+=1;

        #ُSince color value may change, we should update mouse layer color
        self.m_mouse_layer.pen_color = QColor(color[0],color[1],color[2]);
    
    def delete_layer(self, idx):
        self.scene().removeItem(self.layers[idx]);
        del self.layers[idx];
        self.m_layer_count-=1;
        pass
    
    '''
        Clear layers on the image as well as the background radiograph item
    '''
    def clear_whole(self):
        self.scene().clear();
        self.background_item = QtWidgets.QGraphicsPixmapItem();
        
        self.m_mouse_layer = MouseLayer(self.background_item);
        color = QtGui.QColor(255,255,255);
        self.m_mouse_layer.pen_color = color;
        self.m_mouse_layer.pen_thickness = 150;
        self.m_mouse_layer.setZValue(0);
        self.m_mouse_layer.current_state = self.m_current_state;
        self.m_mouse_layer.pen_thickness = self.m_brush_size;
        self.m_mouse_layer.setOpacity(self.m_mouse_layer_opacity);
        
        self.scene().addItem(self.background_item);

        self.m_mouse_layer.grabMouse();
        self.m_mouse_layer.grabKeyboard();
        
        self.layers.clear();
        self.m_active_layer = -1;
        self.m_layer_count = 0;
        
    '''
        Clear layers on the image.
        Background image will remain.
    '''
    def clear_layers(self):
        for l in range(len(self.layers)):
            self.scene().removeItem(self.layers[l]);
        #del self.layers[:];
        self.layers.clear();
        self.m_active_layer = -1;
        self.m_layer_count = 0;

        #reset mouse layer state
        color = QtGui.QColor(255,255,255);
        self.m_mouse_layer.pen_color = color;
        self.m_mouse_layer.pen_thickness = 150;
        self.m_mouse_layer.setZValue(0);
        self.m_mouse_layer.current_state = self.m_current_state;
        self.m_mouse_layer.pen_thickness = self.m_brush_size;
        self.m_mouse_layer.setOpacity(self.m_mouse_layer_opacity);

    def wheelEvent(self, event):
        if not self.m_empty:
            scene_pos = self.mapToScene(event.pos());
            if event.angleDelta().y() > 0:
                factor = 1.25
                self.m_zoom += 1
            else:
                factor = 0.8
                self.m_zoom -= 1
            
            if self.m_zoom > 0:
                self.scale(factor, factor);
                self.centerOn(scene_pos);
            elif self.m_zoom == 0:
               self.fitInView(self.background_item, QtCore.Qt.KeepAspectRatio)
            else:
                self.m_zoom = 0
      
    def mousePressEvent(self, event):

        #Update mouse buttons state
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            self.m_right_mouse_down = True;
        elif event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.m_left_mouse_down = True;
        elif event.button() == QtCore.Qt.MouseButton.MidButton:
            self.m_mid_mouse_down = True;
        #--------------------------------------------------

        if self.background_item.isUnderMouse() and self.m_right_mouse_down:
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor);
            event.accept();
            
        super(RadiographViewer, self).mousePressEvent(event)
    
    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):

        #Update mouse buttons state
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            self.m_right_mouse_down = False;
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor);
            event.accept();

        elif event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.m_left_mouse_down = False;
        elif event.button() == QtCore.Qt.MouseButton.MidButton:
            self.m_mid_mouse_down = False;
        #--------------------------------------------------
        super(RadiographViewer, self).mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        
        scene_pos = self.mapToScene(event.pos());
        if self.m_right_mouse_down == True:
            offset_x = event.pos().x() - self.m_last_x_view;
            offset_y = event.pos().y() - self.m_last_y_view;
            
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - offset_x);
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - offset_y);
        
        #update mouses last position
        self.m_last_y_view = event.pos().y();
        self.m_last_x_view = event.pos().x();
        #----------------------------------------------------
        if self.background_item.isUnderMouse():
            #print(self.mapToScene(event.pos()));
            self.coordinate_label.setText(f"Mouse Position ({int(scene_pos.x())},{int(scene_pos.y())})");

        if self.m_mouse_layer is not None:
            self.m_mouse_layer.update_cursor(scene_pos);

        super(RadiographViewer, self).mouseMoveEvent(event)
    
    def keyPressEvent(self, event: QtGui.QKeyEvent):
        #Increament pen size
        if event.key() == QtCore.Qt.Key.Key_A:
            if self.m_brush_size < Config.MAX_PEN_SIZE:
                self.m_brush_size = self.m_brush_size + 1;
                self.size_changed_signal.emit(self.m_brush_size);
        
        if event.key() == QtCore.Qt.Key.Key_Z:
            if self.m_mouse_layer.pen_thickness > 0:
                self.m_brush_size = self.m_brush_size - 1;
                self.size_changed_signal.emit(self.m_brush_size);

        super(RadiographViewer, self).keyPressEvent(event);

#------------------------------------------------------------------