#==================================================================
#==================================================================
from copy import deepcopy
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QColor, QCursor, QImage, QPixmap, QPainter, QPen
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsPixmapItem
import numpy as np
from numpy.core.fromnumeric import size
from LayerItems import *
import math
import Config
from utils import pixmap_to_numpy
#==================================================================
#==================================================================

class RadiographViewer(QtWidgets.QGraphicsView):
    size_changed_signal = QtCore.pyqtSignal(int);
    def __init__(self, parent):
        super().__init__(parent);
        
        self.m_zoom = 0;
        self.m_empty = True;
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
        self.num_constant_layer = 3;
        self.coordinate_label = None;

        self.rect_start = None;
        self.rect_end = None;

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
        self.m_mouse_layer.setZValue(0);
        
        self.scene().addItem(self.background_item);

        self.shortcut_undo = QShortcut(QKeySequence('Ctrl+Z'), self);
        self.shortcut_undo.activated.connect(self.undo_event);

        self.shortcut_redo = QShortcut(QKeySequence('Ctrl+Y'), self);
        self.shortcut_redo.activated.connect(self.redo_event);

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
                self.layers[self.m_active_layer].background_gc_layer.pen_thickness = size;
                self.layers[self.m_active_layer].foreground_gc_layer.pen_thickness = size;
        self.m_mouse_layer.update();

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
    
    def set_active_layer_index(self, idx):
        self.m_active_layer = idx;

    def get_active_layer_index(self):
        return self.m_active_layer;
    
    def get_layer_names(self):
        ret = [];
        for l in self.layers:
            ret.append(l.name);
        return ret;

    def set_state(self, state):
        self.m_current_state = state;
        if self.m_active_layer != -1 and self.m_active_layer < self.m_layer_count:
            self.layers[self.m_active_layer].current_state = state;
            self.layers[self.m_active_layer].background_gc_layer.current_state = state;
            self.layers[self.m_active_layer].foreground_gc_layer.current_state = state;
        
        #set state of the mouse layer
        self.m_mouse_layer.current_state = state;
    
    def get_state(self):
        return self.m_current_state;

    def set_layer_pixmap(self, idx, pixmap):
        self.layers[idx].m_pixmap = pixmap;
        pass

    def set_image(self, pixmap=None, reset = True):
        self.__current_image = pixmap.toImage();
        self.__current_pixmap = pixmap;
        self.background_item.setPixmap(pixmap);
        
        #Reset each layer lasso tool by new given image
        for i in range(self.m_layer_count):
            self.layers[i].reset_lasso_tool(pixmap);
        
        if reset:
            self.pixels = pixmap_to_numpy(pixmap);
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
    def set_layer_visibility(self, b, idx = None):
        if idx is not None:
            self.layers[idx].visible = b;
            self.layers[idx].update();
        else:
            if self.m_active_layer != -1 and self.m_active_layer < self.m_layer_count:
                self.layers[self.m_active_layer].visible = b;
                self.layers[self.m_active_layer].update();
    
    def set_gc_layers_visibility(self, b, idx = None):
        if idx is not None:
            self.layers[idx].background_gc_layer.visible = b;
            self.layers[idx].foreground_gc_layer.visible = b;
            #self.update();
            #self.background_item.update();
            self.layers[idx].foreground_gc_layer.update();
            self.layers[idx].background_gc_layer.update();
        else:
            if self.m_active_layer != -1 and self.m_active_layer < self.m_layer_count:
                self.layers[self.m_active_layer].background_gc_layer.visible = b;
                self.layers[self.m_active_layer].foreground_gc_layer.visible = b;
                self.layers[idx].foreground_gc_layer.update();
                self.layers[idx].background_gc_layer.update();
                #self.update();

    '''
        Sets the opacity of the current active layer to the given value
    '''
    def set_layer_opacity(self, v):
        if self.m_active_layer != -1 and self.m_active_layer < self.m_layer_count:
            self.layers[self.m_active_layer].setOpacity(v/255);
            self.layers[self.m_active_layer].background_gc_layer.setOpacity(v/255);
            self.layers[self.m_active_layer].foreground_gc_layer.setOpacity(v/255);
    
    '''
        Type 1 for background and tpye 0 for foreground.
        If type is none we do it for both
    '''
    def set_gc_layer_active(self, a, idx=None, type=None):
        if self.m_active_layer != -1 and self.m_active_layer < self.m_layer_count:
            if idx is None:
                idx = self.m_active_layer;

            self.layers[idx].foreground_gc_layer.pen_thickness = self.m_brush_size;
            self.layers[idx].background_gc_layer.pen_thickness = self.m_brush_size;
            self.layers[idx].foreground_gc_layer.current_state = self.m_current_state;
            self.layers[idx].background_gc_layer.current_state = self.m_current_state;

            if type is None:
                self.layers[idx].background_gc_layer.active = a;
                self.layers[idx].foreground_gc_layer.active = a;

                if a is False:
                    self.layers[self.m_active_layer].background_gc_layer.ungrabMouse();
                    self.layers[self.m_active_layer].foreground_gc_layer.ungrabMouse();

                    self.layers[self.m_active_layer].background_gc_layer.ungrabKeyboard();
                    self.layers[self.m_active_layer].foreground_gc_layer.ungrabKeyboard();

            elif type == 1:
                self.layers[idx].background_gc_layer.active = a;
                    #self.layers[self.m_active_layer].background_gc_layer.setEnabled(a);
                    #self.layers[self.m_active_layer].background_gc_layer.setActive(a);
                if a is True:
                    self.layers[idx].background_gc_layer.grabMouse();
                    self.layers[idx].background_gc_layer.grabKeyboard();
                    self.m_mouse_layer.pen_color = self.layers[idx].foreground_gc_layer.pen_color;
                elif a is False:
                    self.layers[idx].background_gc_layer.ungrabMouse();
                    self.layers[idx].background_gc_layer.ungrabKeyboard();

            elif type == 0:
                self.layers[idx].foreground_gc_layer.active = a;
                #self.layers[self.m_active_layer].foreground_gc_layer.setEnabled(a);
                #self.layers[self.m_active_layer].foreground_gc_layer.setActive(a);
                if a is True:
                    self.layers[idx].foreground_gc_layer.grabMouse();
                    self.layers[idx].foreground_gc_layer.grabKeyboard();
                    self.m_mouse_layer.pen_color = self.layers[idx].foreground_gc_layer.pen_color;
                elif a is False:
                    self.layers[idx].foreground_gc_layer.ungrabMouse();
                    self.layers[idx].foreground_gc_layer.ungrabKeyboard();
    
    '''
        Sets the opacity of mouse layer
    '''
    def set_mouse_layer_opacity(self, v):
        self.m_mouse_layer_opacity = v/255;
        self.m_mouse_layer.setOpacity(self.m_mouse_layer_opacity );
        pass
    
    def deactive_all_layers(self):
        for i in range(self.m_layer_count):
            self.set_layer_active(False, i);
    
    def hide_all_layers(self):
        for i in range(self.m_layer_count):
            self.set_layer_visibility(False, i);
    
    def set_layer_active(self, b, idx=None):
        if self.m_active_layer < self.m_layer_count and self.m_active_layer != -1:
            if idx is not None:
                #self.layers[self.m_active_layer].ungrabMouse();
                #self.layers[self.m_active_layer].ungrabKeyboard();
                #self.layers[self.m_active_layer].active = False;
                #self.set_gc_layer_active(False, self.m_active_layer);
                #self.layers[self.m_active_layer].setEnabled(False);
                #self.layers[self.m_active_layer].setActive(False);

                self.layers[idx].active = b;
                #self.layers[idx].setEnabled(b);
                #self.layers[idx].setActive(b);
                # a = self.mouseGrabber;
                # if a == self.layers[idx]:
                #     print("Already!");
                if b == True: 
                    self.layers[self.m_active_layer].grabMouse();
                    self.layers[self.m_active_layer].grabKeyboard();
                else:
                    self.layers[self.m_active_layer].ungrabMouse();
                    self.layers[self.m_active_layer].ungrabKeyboard();

                self.layers[idx].pen_thickness = self.m_brush_size;
                self.layers[idx].current_state = self.m_current_state;

                # self.layers[idx].foreground_gc_layer.pen_thickness = self.m_brush_size;
                # self.layers[idx].background_gc_layer.pen_thickness = self.m_brush_size;

                # self.layers[idx].foreground_gc_layer.current_state = self.m_current_state;
                # self.layers[idx].background_gc_layer.current_state = self.m_current_state;

                #ُSince color value may change, we should update mouse layer color
                if b == True:
                    self.m_mouse_layer.pen_color = self.layers[idx].pen_color;
                self.layers[idx].foreground_gc_layer.pen_color = self.layers[idx].pen_color;
                self.layers[idx].background_gc_layer.pen_color = QColor(255-self.layers[idx].pen_color.red(), 255-self.layers[idx].pen_color.green(),255-self.layers[idx].pen_color.blue());
            #-----------------------------------------------------
            else:
                self.layers[self.m_active_layer].active = b;
                #self.layers[self.m_active_layer].setEnabled(b);
                #self.layers[self.m_active_layer].setActive(b);
                if b is True:
                    self.layers[self.m_active_layer].grabMouse();
                    self.layers[self.m_active_layer].grabKeyboard();
                    self.m_mouse_layer.pen_color = self.layers[self.m_active_layer].pen_color;

    def add_layer(self, name, color):
        foreground_item = LayerItem(self.background_item);
        foreground_item.pen_color = QColor(color[0],color[1],color[2]);
        foreground_item.pen_thickness = self.m_brush_size;
        foreground_item.current_state = self.m_current_state;
        foreground_item.setZValue(self.m_layer_count+1)
        foreground_item.reset();
        foreground_item.reset_lasso_tool(self.__current_pixmap);
        foreground_item.name = name;

        #Add foreground and background of grabcut for this layer
        background_gc_layer = LayerItem(self.background_item);
        background_gc_layer.pen_thickness = self.m_brush_size;
        background_gc_layer.current_state = self.m_current_state;
        background_gc_layer.setZValue(self.m_layer_count+1)
        background_gc_layer.reset();

        foreground_gc_layer = LayerItem(self.background_item);
        foreground_gc_layer.pen_thickness = self.m_brush_size;
        foreground_gc_layer.current_state = self.m_current_state;
        foreground_gc_layer.setZValue(self.m_layer_count+1)
        foreground_gc_layer.reset();

        foreground_item.foreground_gc_layer = foreground_gc_layer;
        foreground_item.background_gc_layer = background_gc_layer;
        #-------------------------------------------------------

        #set mouse layer as the top layer
        self.m_mouse_layer.setZValue(self.m_layer_count+3);
        
        if self.m_active_layer < self.m_layer_count and self.m_active_layer != -1:
            self.layers[self.m_active_layer].active = False;
            self.set_gc_layer_active(False, idx = self.m_active_layer);
            #self.layers[self.m_active_layer].ungrabKeyboard();

        #deactive other layers and active currently added layer
        # for l in self.layers:
        #     l.active = False;
        #     l.setEnabled(False);
        #     l.setActive(False);
        foreground_item.active = True;
        #foreground_item.setEnabled(True);
        #foreground_item.setActive(True);
        foreground_item.grabMouse();
        foreground_item.grabKeyboard();
    
        self.layers.append(foreground_item);
        self.m_active_layer = len(self.layers)-1;
        self.m_layer_count+=1;

        #ُSince color value may change, we should update mouse layer color
        self.m_mouse_layer.pen_color = QColor(color[0],color[1],color[2]);
        foreground_item.foreground_gc_layer.pen_color = QColor(color[0],color[1],color[2]);
        foreground_item.background_gc_layer.pen_color = QColor(255-color[0],255-color[1],255-color[2]);
    
    def delete_layer(self, idx):
        self.scene().removeItem(self.layers[idx]);
        self.scene().removeItem(self.layers[idx].background_gc_layer);
        self.scene().removeItem(self.layers[idx].foreground_gc_layer);
        del self.layers[idx];
        self.m_layer_count-=1;

        #set mouse layer as the top layer
        self.m_mouse_layer.setZValue(self.m_layer_count + 3);
        
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
        #self.foreground_gc_layer.grabMouse();
        #self.background_gc_layer.grabMouse();
        
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
    
    def get_numpy(self):
        image = self.background_item.toImage()
        s = image.bits().asstring(image.width() * image.height() * 4)
        arr = np.fromstring(s, dtype=np.uint8).reshape((image.height(), image.width(), 4)) 
        return arr;

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
        scene_pos = self.mapToScene(event.pos());
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
                self.m_mouse_layer.update();
        
        if event.key() == QtCore.Qt.Key.Key_Z:
            if self.m_mouse_layer.pen_thickness > 0:
                self.m_brush_size = self.m_brush_size - 1;
                self.size_changed_signal.emit(self.m_brush_size);
            
        super(RadiographViewer, self).keyPressEvent(event);
    
    def undo_event(self):
        if self.m_active_layer != -1 and self.m_active_layer < self.m_layer_count:
            self.layers[self.m_active_layer].undo_event();
            self.layers[self.m_active_layer].foreground_gc_layer.undo_event();
            self.layers[self.m_active_layer].background_gc_layer.undo_event();
    
    def redo_event(self):
        if self.m_active_layer != -1 and self.m_active_layer < self.m_layer_count:
            self.layers[self.m_active_layer].redo_event();
            self.layers[self.m_active_layer].foreground_gc_layer.redo_event();
            self.layers[self.m_active_layer].background_gc_layer.redo_event();

    def foreground_clicked_slot(self):
        if self.m_active_layer < self.m_layer_count and self.m_active_layer != -1:
            if self.layers[self.m_active_layer].foreground_gc_layer.active is False:
                #self.layers[self.m_active_layer].background_gc_layer.setActive(False);
                #self.layers[self.m_active_layer].background_gc_layer.setEnabled(False);
                self.layers[self.m_active_layer].background_gc_layer.active = False;
                #self.layers[self.m_active_layer].background_gc_layer.ungrabMouse();
                
                #self.layers[self.m_active_layer].foreground_gc_layer.setActive(True);
                #self.layers[self.m_active_layer].foreground_gc_layer.setEnabled(True);
                self.layers[self.m_active_layer].foreground_gc_layer.active = True;
                self.layers[self.m_active_layer].foreground_gc_layer.grabMouse();
                self.layers[self.m_active_layer].foreground_gc_layer.grabKeyboard();

                self.set_state(LayerItem.DrawState);
            else:
                #self.layers[self.m_active_layer].background_gc_layer.setActive(False);
                #self.layers[self.m_active_layer].background_gc_layer.setEnabled(False);
                self.layers[self.m_active_layer].background_gc_layer.active = False;
                #self.layers[self.m_active_layer].background_gc_layer.ungrabMouse();
                
                #self.layers[self.m_active_layer].foreground_gc_layer.setActive(False);
                #self.layers[self.m_active_layer].foreground_gc_layer.setEnabled(False);
                self.layers[self.m_active_layer].foreground_gc_layer.active = False;
                #self.layers[self.m_active_layer].foreground_gc_layer.ungrabMouse();

    def background_clicked_slot(self):
        if self.m_active_layer < self.m_layer_count and self.m_active_layer != -1:
            if self.layers[self.m_active_layer].background_gc_layer.active is False:
                #self.layers[self.m_active_layer].foreground_gc_layer.setActive(False);
                #self.layers[self.m_active_layer].foreground_gc_layer.setEnabled(False);
                self.layers[self.m_active_layer].foreground_gc_layer.active = False;
                #self.layers[self.m_active_layer].foreground_gc_layer.ungrabMouse();

                #self.layers[self.m_active_layer].background_gc_layer.setActive(True);
                #self.layers[self.m_active_layer].background_gc_layer.setEnabled(True);
                self.layers[self.m_active_layer].background_gc_layer.active = True;
                self.layers[self.m_active_layer].background_gc_layer.grabMouse();
                self.layers[self.m_active_layer].background_gc_layer.grabKeyboard();

                self.set_state(LayerItem.DrawState);
            else:

                #self.layers[self.m_active_layer].foreground_gc_layer.setActive(False);
                #self.layers[self.m_active_layer].foreground_gc_layer.setEnabled(False);
                self.layers[self.m_active_layer].foreground_gc_layer.active = False;
                #self.layers[self.m_active_layer].foreground_gc_layer.ungrabMouse();

                #self.layers[self.m_active_layer].background_gc_layer.setActive(False);
                #self.layers[self.m_active_layer].background_gc_layer.setEnabled(False);
                self.layers[self.m_active_layer].background_gc_layer.active = False;
                #self.layers[self.m_active_layer].background_gc_layer.ungrabMouse();

    
    def reset_gc_slot(self):
        if self.m_active_layer < self.m_layer_count and self.m_active_layer != -1:
            self.layers[self.m_active_layer].reset_mask_gc();
    
    def update_foreground_with_layer_slot(self):
        if self.m_active_layer < self.m_layer_count and self.m_active_layer != -1:
            self.layers[self.m_active_layer].foreground_gc_layer.undostack.append(self.layers[self.m_active_layer].foreground_gc_layer.m_pixmap.copy(QRect()));
            self.layers[self.m_active_layer].foreground_gc_layer.m_pixmap = self.layers[self.m_active_layer].m_pixmap.copy();
            self.layers[self.m_active_layer].foreground_gc_layer.update();
      
    def update_gc_slot(self):
        if self.m_active_layer < self.m_layer_count and self.m_active_layer != -1:
            img_arr = np.fromstring(self.__current_image.bits().asstring(self.__current_image.width() * self.__current_image.height() * 4), 
            dtype=np.uint8).reshape((self.__current_image.height(), self.__current_image.width(), 4))
            img_arr = img_arr[:,:,:3]; 

            bkgr = self.layers[self.m_active_layer].background_gc_layer.get_numpy();
            frgr = self.layers[self.m_active_layer].foreground_gc_layer.get_numpy();

            #Check if any pixel is marked as foreground
            bkgr = bkgr[:,:,:3];
            frgr = frgr[:,:,:3];
            #cv2.imshow("foreground", frgr);
            #cv2.imshow("background", bkgr);
            #cv2.waitKey();
            frgr = np.sum(frgr, axis=2, dtype=np.int32);
            bkgr = np.sum(bkgr, axis=2, dtype=np.int32);
            pixels_marked_foregound = frgr[frgr != 0];
            pixels_marked_foregound = pixels_marked_foregound.sum();

            if pixels_marked_foregound !=0:
                current_color = self.layers[self.m_active_layer].pen_color;
                current_color_sum = current_color.red() + current_color.green() + current_color.blue();

                frgr[frgr== current_color_sum] = 1;
                bkgr[bkgr== (3*255)-current_color_sum] = 1;
                frgr = frgr.astype("uint8");
                bkgr = bkgr.astype("uint8");
                
                foreground_from_mask = np.where((self.layers[self.m_active_layer].mask_gc == cv2.GC_FGD) | (self.layers[self.m_active_layer].mask_gc == cv2.GC_PR_FGD), 1,0).astype("uint8");
                foreground_excluded = (frgr == 0);
                foreground_from_mask[foreground_from_mask == True] = 1;
                foreground_excluded[foreground_excluded == True] = 1;
                foreground_from_mask = np.array(foreground_from_mask, dtype='uint8');
                foreground_excluded = np.array(foreground_excluded, dtype='uint8');

                background_new_pix = np.where((foreground_from_mask==1) & (foreground_excluded == 1), 1,0).astype("uint8");

                self.layers[self.m_active_layer].mask_gc[background_new_pix==1] = 0;

                background_from_mask = np.where((self.layers[self.m_active_layer].mask_gc== cv2.GC_BGD) | (self.layers[self.m_active_layer].mask_gc == cv2.GC_PR_BGD), 1,0).astype("uint8");
                foreground_included = (frgr == 1);
                background_from_mask[background_from_mask == True] = 1;
                foreground_included[foreground_included == True] = 1;
                background_from_mask = np.array(background_from_mask, dtype='uint8');
                foreground_included = np.array(foreground_included, dtype='uint8');
                foreground_new_pix = np.where((background_from_mask==1) & (foreground_included == 1), 1,0).astype("uint8");

                self.layers[self.m_active_layer].mask_gc[foreground_new_pix==1] = cv2.GC_FGD;

                self.layers[self.m_active_layer].mask_gc[bkgr ==1] = cv2.GC_BGD;

                bgdModel = np.zeros((1,65),np.float64)
                fgdModel = np.zeros((1,65),np.float64)

                (self.layers[self.m_active_layer].mask_gc, bgModel, fgModel) = cv2.grabCut(img_arr, self.layers[self.m_active_layer].mask_gc, None, bgdModel,
                        fgdModel, iterCount=5, mode=cv2.GC_INIT_WITH_MASK);
                    
                output_mask = np.where((self.layers[self.m_active_layer].mask_gc == cv2.GC_BGD) | (self.layers[self.m_active_layer].mask_gc == cv2.GC_PR_BGD), 0, 1).astype("uint8");
                self.mask_pixmap = output_mask;
                binar = (output_mask==1);
                output_mask = np.repeat(np.expand_dims(output_mask, axis=2), 3,axis=2);
                output_mask[binar==True] = np.array([current_color.red(), current_color.green(), current_color.blue()]).astype("uint8");

                height,width,_ = output_mask.shape;
                bytesPerLine = 3 * width
                qImg = QImage(output_mask.data, width, height, bytesPerLine, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qImg);

                mask = pixmap.createMaskFromColor(QtGui.QColor(0,0,0),Qt.MaskMode.MaskInColor);
                pixmap.setMask(mask);

                #Add current data to undo stack 
                self.layers[self.m_active_layer].foreground_gc_layer.undostack.append(self.layers[self.m_active_layer].foreground_gc_layer.m_pixmap.copy(QRect()));
                self.layers[self.m_active_layer].background_gc_layer.undostack.append(self.layers[self.m_active_layer].background_gc_layer.m_pixmap.copy(QRect()));
                self.layers[self.m_active_layer].undostack.append(self.layers[self.m_active_layer].m_pixmap.copy(QRect()));

                self.layers[self.m_active_layer].background_gc_layer.clear();
                self.layers[self.m_active_layer].foreground_gc_layer.clear();
                
                self.set_layer_pixmap(self.m_active_layer, pixmap);
                self.layers[self.m_active_layer].foreground_gc_layer.m_pixmap = pixmap.copy(QRect());
                self.layers[self.m_active_layer].foreground_gc_layer.update();


#------------------------------------------------------------------