#==================================================================
#==================================================================
from copy import deepcopy
from posixpath import basename
from PIL import ImageColor
from PyQt5.QtCore import QThread, Qt
from PyQt5.QtWidgets import QApplication, QCheckBox, QColorDialog, QComboBox, QDesktopWidget, QGridLayout, QGroupBox, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QProgressBar, QPushButton, QRadioButton, QSlider, QFileDialog, QDialog, QStatusBar, QTextEdit, QVBoxLayout
import sys
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from RadiographViewer import *
from DataPoolHandler import *
from Utility import *
from NetworkTrainer import *
from ProjectHandler import *
from PIL.ImageColor import *
import Config
import logging
import traceback
#==================================================================
#==================================================================

class WaitingWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowModality(Qt.ApplicationModal)
        self.title = "Processing...";
        self.left = 0;
        self.top = 0;
        self.width = 450;
        self.height = 50;
        self.init_ui();
        self.layers_selected = [];
        

    def init_ui(self):
        centerPoint = QDesktopWidget().availableGeometry().center();
        self.setGeometry(centerPoint.x(), centerPoint.y(), self.width, self.height);
        self.setWindowTitle(self.title);
        self.setFixedSize(self.width, self.height);

        self.window_grid_layout = QGridLayout(self);

        item_count = 0;

        self.progress_bar_epoch = QProgressBar();
        self.progress_bar_epoch.setValue(0);
        self.progress_bar_epoch.setMinimum(0);
        self.progress_bar_epoch.setMaximum(0)
        self.window_grid_layout.addWidget(self.progress_bar_epoch,item_count,0);
        item_count+=1;
    
    def closeEvent(self, evnt):
        evnt.ignore()

class TrainingInfoWindow(QWidget):
    terminate_signal = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setWindowModality(Qt.ApplicationModal)
        self.title = "Training info";
        self.left = 0;
        self.top = 0;
        self.width = 500;
        self.height = 500;
        self.init_ui();
        self.layers_selected = [];

    def init_ui(self):

        centerPoint = QDesktopWidget().availableGeometry().center();
        self.setGeometry(centerPoint.x() - self.width/2, centerPoint.y()-self.height/2, self.width, self.height);
        self.setWindowTitle(self.title);

        self.window_grid_layout = QGridLayout(self);
        self.window_grid_layout.setContentsMargins(20,20,20,20);

        item_count = 0;

        self.current_epoch = QLabel('Current epoch: 0');
        self.window_grid_layout.addWidget(self.current_epoch,item_count,0,1,1);
        item_count+=1;
        
        self.train_label = QLabel('Train iteration progress');
        self.window_grid_layout.addWidget(self.train_label,item_count,0,1,1);
        item_count+=1;

        self.progress_bar_epoch = QProgressBar();
        self.progress_bar_epoch.setValue(0);
        self.progress_bar_epoch.setMinimum(0);
        self.progress_bar_epoch.setMaximum(0)
        self.window_grid_layout.addWidget(self.progress_bar_epoch,item_count,0,1,1);
        item_count+=1;

        self.valid_label = QLabel('valid iteration progress');
        self.window_grid_layout.addWidget(self.valid_label,item_count,0,1,1);
        item_count+=1;

        self.progress_bar_epoch_valid = QProgressBar();
        self.progress_bar_epoch_valid.setValue(0);
        self.progress_bar_epoch_valid.setMinimum(0);
        self.progress_bar_epoch_valid.setMaximum(100)
        self.window_grid_layout.addWidget(self.progress_bar_epoch_valid,item_count,0,1,1);
        item_count+=1;

        self.textbox_info = QTextEdit("Waiting for data augmentation to finish...");
        self.textbox_info.setReadOnly(True);
        self.window_grid_layout.addWidget(self.textbox_info,item_count,0,1,2);
        item_count+=1;
        
        # self.stop_button = QPushButton("Stop");
        # self.stop_button.clicked.connect(self.terminate_training);
        # self.window_grid_layout.addWidget(self.stop_button,item_count,0,1,2);
    
    def update_train_info_iter_slot(self, val):
        self.progress_bar_epoch.setValue(val*100);
        self.update();
    
    def update_valid_info_iter_slot(self, val):
        self.progress_bar_epoch_valid.setValue(val*100);
        self.update();
    
    def update_train_info_epoch_train_slot(self, col, val, time, epoch):
        self.current_epoch.setText(f"Current epoch: {epoch}");
        txt = f"Epoch {epoch} completed in {int(time)} seconds\n Training results: ";
        for name, value in zip(col, val):
                if(name != 'cm'):
                    txt += f" | {name}: {value}"
        self.textbox_info.append(txt);
    
    def update_train_info_epoch_valid_slot(self, col, val):
        txt = "Validation results: \n";
        for name, value in zip(col, val):
                if(name != 'cm'):
                    #self.writer.add_scalar('training/' + name,float(value),engine.state.epoch);
                    txt += f"{name}: {value} | "
        txt += "\n";
        self.textbox_info.append(txt);
    
    def augmentation_finished_slot(self):
        self.textbox_info.append("Augmentation finshed, start training...");
        self.progress_bar_epoch.setMaximum(100)
    
    def terminate_training(self):
        Config.FINISH_TRAINING = True;
        #self.terminate_signal.emit();

    def show_window(self):
        self.textbox_info.clear();
        self.textbox_info.setText("Waiting for data augmentation to finish...");
        self.current_epoch.setText("Current epoch: 0");
        self.progress_bar_epoch.setValue(0);
        self.progress_bar_epoch.setMaximum(0)
        self.adjustSize();
        self.show();

class SegmentationOptionsWindow(QWidget):
    confirm_clicked_signal = pyqtSignal(str, str)
    
    def __init__(self):
        super().__init__()
        self.setWindowModality(Qt.ApplicationModal)
        self.title = "Options";
        self.left = 0;
        self.top = 0;
        self.width = 500;
        self.height = 500;
        self.current_color = QtGui.QColor(0, 0, 0)
        self.color_selected = "black";
        self.name_selected = Config.PREDEFINED_NAMES[0];
        
        self.init_ui();

    def init_ui(self):
        centerPoint = QDesktopWidget().availableGeometry().center();
        self.setGeometry(centerPoint.x() - self.width/2, centerPoint.y() - self.height/2, self.width, self.height);
        self.setWindowTitle(self.title);
        self.window_grid_layout = QGridLayout(self);
        self.window_grid_layout.setContentsMargins(20,20,20,20);
        self.box_gridLayout = QtWidgets.QGridLayout()
        self.name_gorupbox = QGroupBox();
        
        self.custom_name = QRadioButton('Custom Name:');
        self.custom_name.toggled.connect(lambda:self.btnstate(self.custom_name))
        
        self.segmentation_name = QLineEdit();
        self.segmentation_name.setEnabled(False);
        
        self.pen_label = QLabel('Pen color: ');
        
        self.pen_button = QPushButton(clicked = self.show_color_dlg);
        self.pen_button.setStyleSheet(
            "background-color: {}".format(self.current_color.name())
        )

        self.confirm_button = QPushButton('Confirm');
        self.confirm_button.clicked.connect(self.confirm_clicked);
        
        #Row contraction
        self.verticalSpacer = QtWidgets.QSpacerItem(20, 40,
         QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding) 
        #--------------------------------------------------------------

        self.hide();

    def update_ui(self):
        for i in reversed(range(self.box_gridLayout.count())): 
            widgetToRemove = self.box_gridLayout.itemAt(i).widget();
            self.box_gridLayout.removeWidget(widgetToRemove);
            widgetToRemove.setParent(None);
        
        for i in reversed(range(self.window_grid_layout.count())): 
            widgetToRemove = self.window_grid_layout.itemAt(i).widget();
            self.window_grid_layout.removeWidget(widgetToRemove);
            widgetToRemove.setParent(None);

        num_items = 0;
        for n in Config.PROJECT_PREDEFINED_NAMES[0]:
            rbtn = QRadioButton(n);
            if num_items == 0:
                rbtn.setChecked(True);
            rbtn.toggled.connect(lambda:self.btnstate(rbtn))
            self.box_gridLayout.addWidget(rbtn, num_items, 0, 1, 1);
            num_items+=1;

        
        self.box_gridLayout.addWidget(self.custom_name,num_items,0,1,1);

        self.box_gridLayout.addWidget(self.segmentation_name,num_items,1,1,1);
        num_items+=1;

        self.box_gridLayout.addWidget(self.pen_label,num_items,0,1,1);

        self.box_gridLayout.addWidget(self.pen_button,num_items,1,1,1);
        num_items+=1;

        self.box_gridLayout.addWidget(self.confirm_button,num_items,0,1,2);

        self.name_gorupbox.setLayout(self.box_gridLayout);
        self.window_grid_layout.addWidget(self.name_gorupbox,0,0,num_items,1);

        #self.window_grid_layout.addWidget(self.verticalSpacer, num_items, 0);

        self.adjustSize();

    def show_window(self, layers_name, color):
        self.update_ui();
        self.current_color = color;
        self.segmentation_name.clear();
        self.layers_name = layers_name;
        self.show();

    def hide_window(self):
        self.hide();
    
    def btnstate(self,b):
        rbtn = self.sender();
        if rbtn.text() == 'Custom Name:':
            if self.custom_name.isChecked() == True:
                self.segmentation_name.setEnabled(True);
            elif self.custom_name.isChecked() == False:
                self.segmentation_name.setEnabled(False);
        else:
            self.name_selected = rbtn.text();
    
    def show_color_dlg(self):
        color = QtWidgets.QColorDialog.getColor(self.current_color)
        self.pen_button.setStyleSheet(
            "background-color: {}".format(color.name())
        )
        self.color_selected = color.name();
    
    def confirm_clicked(self):
        if self.custom_name.isChecked() == True:
            self.name_selected = self.segmentation_name.text();
        #Check if the selected layer already exists in list or not.
        found = False;
        for l in self.layers_name:
            if l == self.name_selected:
                found = True;
                break;
        if found:
            show_dialoge(QMessageBox.Icon.Critical, "Layer already exists, use another layer", "Layer exists", QMessageBox.Ok);
        else:
            self.hide();
            self.confirm_clicked_signal.emit(self.name_selected, self.color_selected);

class LayerSelectionWindow(QWidget):
    confirm_clicked_signal = pyqtSignal(list)
    
    def __init__(self):
        super().__init__()
        self.setWindowModality(Qt.ApplicationModal)
        self.title = "Layers for training";
        self.left = 0;
        self.top = 0;
        self.width = 500;
        self.height = 500;
        self.init_ui();
        self.layers_selected = [];

    def init_ui(self):
        centerPoint = QDesktopWidget().availableGeometry().center();
        self.setGeometry(centerPoint.x(), centerPoint.y(), self.width, self.height);
        self.setWindowTitle(self.title);

        self.layer_groupbox  = QGroupBox();
        self.box_gridlayout = QGridLayout();
        self.window_grid_layout = QGridLayout(self);
        self.window_grid_layout.setContentsMargins(20,20,20,20);

        self.chbx_name_label = QLabel('Layer name');
        
        self.count_name_label = QLabel('Available');
        
        self.confirm_button = QPushButton('Confirm');
        self.confirm_button.clicked.connect(self.confirm_clicked);
        self.layers_dict = None;

    def show_window(self, layers):

        for i in reversed(range(self.box_gridlayout.count())): 
            widgetToRemove = self.box_gridlayout.itemAt(i).widget()
            self.box_gridlayout.removeWidget(widgetToRemove)
            widgetToRemove.setParent(None)
        
        for i in reversed(range(self.window_grid_layout.count())): 
            widgetToRemove = self.window_grid_layout.itemAt(i).widget()
            self.window_grid_layout.removeWidget(widgetToRemove)
            widgetToRemove.setParent(None)

        self.box_gridlayout.addWidget(self.chbx_name_label, 0, 0, 1, 1);
        self.box_gridlayout.addWidget(self.count_name_label, 0, 1, 1, 1);

        i = 1;
        self.layers_dict = copy.copy(layers);
        for k in layers.keys():
            val = layers[k];
            ckbx = QCheckBox(k);
            self.layers_dict[k] = 0;
            ckbx.toggled.connect(lambda:self.layer_clicked(ckbx));
            ckbx.setChecked(False);
            self.box_gridlayout.addWidget(ckbx, i, 0, 1, 1);
            count_lbl = QLabel(str(val));
            self.box_gridlayout.addWidget(count_lbl, i, 1, 1, 1);
            i += 1;
        
        self.box_gridlayout.addWidget(self.confirm_button,i,0,1,2);
        self.layer_groupbox.setLayout(self.box_gridlayout);
        self.window_grid_layout.addWidget(self.layer_groupbox, 0,0,i+1,2);
        
        self.adjustSize();
        self.show();
    
    def layer_clicked(self, b):
        chbx = self.sender();
        self.layers_dict[chbx.text()] = chbx.isChecked();
    
    def confirm_clicked(self):
        num_selected = 0;
        layers_selected = [];
        for k in self.layers_dict.keys():
            if self.layers_dict[k] == 1:
                layers_selected.append(k);
                num_selected += 1;
        if num_selected != 0:
            #We first determine number of classes for training here
            #Since we always have background layer as the first layer we should add 1
            Config.NUM_CLASSES = num_selected + 1;
            
            self.confirm_clicked_signal.emit(layers_selected);
            self.hide();
        else:
            show_dialoge(QMessageBox.Icon.Critical, f"You should select one layer at least", "Error", QMessageBox.Ok);

class NewProjectWindow(QWidget):
    confirm_new_project_clicked_signal = pyqtSignal(str)
    open_project_signal = pyqtSignal(str);
    
    def __init__(self):
        super().__init__()
        self.setWindowModality(Qt.ApplicationModal)
        self.title = "New project";
        self.left = 0;
        self.top = 0;
        self.width = 500;
        self.height = 500;
        self.init_ui();
        self.layers_selected = [];

    def init_ui(self):
        centerPoint = QDesktopWidget().availableGeometry().center();
        self.setGeometry(centerPoint.x(), centerPoint.y(), self.width, self.height);
        self.setWindowTitle(self.title);

        self.window_grid_layout = QGridLayout(self);
        self.window_grid_layout.setContentsMargins(20,20,20,20);
        self.label = QLabel("Recent project could not be found. Do you want to create a new project or open an existing one?");
        self.window_grid_layout.addWidget(self.label,0,0,1,2);

        self.create_new_button = QPushButton('Create new');
        self.create_new_button.clicked.connect(self.create_new_clicked);
        self.window_grid_layout.addWidget(self.create_new_button,1,0,1,1);     


        self.open_existing_button = QPushButton('Open existing');
        self.open_existing_button.clicked.connect(self.open_existing_clicked);
        self.window_grid_layout.addWidget(self.open_existing_button,1,1,1,1);  

    def show_window(self):
        self.adjustSize();
        self.show();
    
    def create_new_clicked(self):
        name = QFileDialog.getSaveFileName(window, 'Save File');
        self.confirm_new_project_clicked_signal.emit(name[0]);
        self.hide();
    def open_existing_clicked(self):
        options = QFileDialog.Options()
        dialog = QFileDialog()
        dialog.setOptions(options)

        dialog.setFilter(dialog.filter() | QtCore.QDir.Hidden)

        dialog.setFileMode(QFileDialog.ExistingFiles)

        dialog.setAcceptMode(QFileDialog.AcceptOpen);
        dialog.setDirectory(QtCore.QDir.currentPath())

        if dialog.exec_() == QDialog.Accepted:
            path = dialog.selectedFiles()[0];
            self.open_project_signal.emit(path);
        self.hide();
    
class MainWindow(QMainWindow):
    start_train_signal = pyqtSignal(list);
    save_project_signal = pyqtSignal(dict, bool);
    save_project_as_signal = pyqtSignal(str);
    load_model_signal = pyqtSignal();
    predict_on_unlabeled_signal = pyqtSignal(str, dict);
    open_project_signal = pyqtSignal(str);
    save_most_recent_project_signal = pyqtSignal();
    update_meta_signal = pyqtSignal(str);
    get_project_names_signal = pyqtSignal();
    confirm_new_project_clicked_signal = pyqtSignal(str)

    def __init__(self, data_pool_handler, network_trainer):
        super().__init__();
        self.setWindowModality(QtCore.Qt.ApplicationModal)

        #Parameters
        self.title = "Active Learning";
        self.left = 0;
        self.top = 0;
        self.width = 1200;
        self.height = 600;
        self.close_eye_icon = "Icons/eye_icon_closed.png";
        self.open_eye_icon = "Icons/eye_icon.png"
        self.select_folder_icon = "Icons/select_folder_icon.png";
        self.select_files_icon = "Icons/select_files_icon.png";
        self.draw_icon = "Icons/draw_icon.png";
        self.erase_icon = "Icons/erase_icon.png";
        self.save_icon = "Icons/save_icon.ico";
        self.save_as_icon = "Icons/save_as_icon.png";
        self.open_project_icon = "Icons/open_project_icon.ico";
        self.new_project_icon = "Icons/new_project_icon.png";
        self.data_pool_handler = data_pool_handler;
        self.network_trainer = network_trainer;
        self.segmentation_options_window = SegmentationOptionsWindow();
        self.layer_selection_window = LayerSelectionWindow();
        self.trainig_info_window = TrainingInfoWindow();
        self.busy_indicator_window = WaitingWindow();
        #---------------------------------------------------------------

        self.th = QThread();
        self.network_trainer.moveToThread(self.th);
        self.th.start();
        #self.th.started.connect(self.network_trainer.start_train_slot);

        self.init_ui();

    def init_ui(self):
        #Find best location
        centerPoint = QDesktopWidget().availableGeometry();
        pos_x = (centerPoint.width() - self.width) / 2;
        pos_y = (centerPoint.height() - self.height) / 2;
        
        self.setGeometry(pos_x, pos_y, self.width, self.height);
        #self.setGeometry(self.left, self.top , self.width, self.height);
        self.setWindowTitle(self.title);

        self.centralwidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralwidget);

        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setContentsMargins(20,20,20,20);

        #MenuBar
        self.menubar = self.menuBar();
        self.file_menu = self.menubar.addMenu("File");
        self.file_menu.addAction('New project', self.new_project_slot);
        self.file_menu.addAction('Open project', self.open_project_slot);
        self.file_menu.addAction('Save project', self.save_clicked_slot);
        self.file_menu.addAction('Save as project', self.save_as_slot);
        self.file_menu.addAction("Select folder",self.select_folder_slot);
        self.file_menu.addAction("Select files",self.select_files_slot);
        #-------------------------------------------------

        #Toolbar
        self.toolbar = self.addToolBar('top_toolbar');
        self.new_project_action = self.toolbar.addAction('New project', self.new_project_slot);
        self.new_project_action.setIcon(QtGui.QIcon(self.new_project_icon));
        self.open_project_action = self.toolbar.addAction('Open project', self.open_project_slot);
        self.open_project_action.setIcon(QtGui.QIcon(self.open_project_icon));
        self.select_folder_action = self.toolbar.addAction("Select folder",self.select_folder_slot);
        self.select_folder_action.setIcon(QtGui.QIcon(self.select_folder_icon));
        self.select_files_action = self.toolbar.addAction("Select files",self.select_files_slot);
        self.select_files_action.setIcon(QtGui.QIcon(self.select_files_icon));
        self.save_action = self.toolbar.addAction("Save",self.save_clicked_slot);
        self.save_action.setIcon(QtGui.QIcon(self.save_icon));
        self.save_as_action = self.toolbar.addAction("Save As",self.save_as_slot);
        self.save_as_action.setIcon(QtGui.QIcon(self.save_as_icon));
        #-------------------------------------------------
        
        #Column 0
        self.box_train_params = QGroupBox();
        self.box_train_params.setTitle("Train parameters")
        self.box_train_grid = QGridLayout();
        items_column1 = 0;   
        
        self.device_label = QLabel(self);
        self.device_label.setText("Device");
        self.box_train_grid.addWidget(self.device_label,1,0,1,1);
        items_column1+=1;

        self.box_train_params.setLayout(self.box_train_grid);
        self.gridLayout.addWidget(self.box_train_params,0,0,items_column1,2);

        self.box_segmentation = QGroupBox();
        self.box_segmentation.setTitle("Segmentation")
        self.segmentation_box_grid_layout = QGridLayout();

        self.box_image_processing = QGroupBox();
        self.box_image_processing.setTitle("Image processing");
        self.box_image_processing_layout = QGridLayout();

        self.add_segmentation_button = QPushButton(self);
        self.add_segmentation_button.setText("Add");
        self.segmentation_box_grid_layout.addWidget(self.add_segmentation_button, 0, 0, 1, 1);

        self.delete_segmentation_button = QPushButton(self);
        self.delete_segmentation_button.setText("Delete");
        self.segmentation_box_grid_layout.addWidget(self.delete_segmentation_button, 0, 1, 1, 1);
        items_column1+=1;

        self.segments_list = QListWidget(self);
        self.segmentation_box_grid_layout.addWidget(self.segments_list,1,0,1,2);
        items_column1+=1;

        self.paint_button = QPushButton();
        self.paint_button.setText("Paint");
        self.paint_button.setStyleSheet("background-color: Aquamarine")
        self.paint_button.setIcon(QtGui.QIcon(self.draw_icon))
        self.segmentation_box_grid_layout.addWidget(self.paint_button, 2, 0, 1, 1);

        self.erase_button = QPushButton();
        self.erase_button.setText("Erase");
        self.erase_button.setStyleSheet("background-color: white")
        self.erase_button.setIcon(QtGui.QIcon(self.erase_icon))
        self.segmentation_box_grid_layout.addWidget(self.erase_button, 2, 1, 1, 1);
        items_column1+=1;

        self.size_label = QLabel(self);
        self.size_label.setText("Brush size: 150");
        self.segmentation_box_grid_layout.addWidget(self.size_label, 3, 0, 1, 1);
        items_column1+=1;

        self.size_slider = QSlider(Qt.Orientation.Horizontal);
        self.size_slider.setMinimum(1);
        self.size_slider.setMaximum(Config.MAX_PEN_SIZE);
        self.size_slider.setValue(150);
        self.segmentation_box_grid_layout.addWidget(self.size_slider, 4,0,1,2);
        items_column1+=1;

        self.opacity_layer_label = QLabel(self);
        self.opacity_layer_label.setText("Layer Opacity: 255");
        self.segmentation_box_grid_layout.addWidget(self.opacity_layer_label, 5, 0, 1, 1);
        items_column1+=1;

        self.opacity_layer_slider = QSlider(Qt.Orientation.Horizontal);
        self.opacity_layer_slider.setMinimum(0);
        self.opacity_layer_slider.setMaximum(255);
        self.opacity_layer_slider.setValue(255);
        self.segmentation_box_grid_layout.addWidget(self.opacity_layer_slider, 6,0,1,2);
        items_column1+=1;

        self.opacity_marker_label = QLabel(self);
        self.opacity_marker_label.setText("Marker Opacity: 255");
        self.segmentation_box_grid_layout.addWidget(self.opacity_marker_label, 7, 0, 1, 1);
        items_column1+=1;

        self.opacity_marker_slider = QSlider(Qt.Orientation.Horizontal);
        self.opacity_marker_slider.setMinimum(0);
        self.opacity_marker_slider.setMaximum(255);
        self.opacity_marker_slider.setValue(255);
        self.segmentation_box_grid_layout.addWidget(self.opacity_marker_slider, 8,0,1,2);
        items_column1+=1;

        self.next_sample_button = QPushButton();
        self.next_sample_button.setText("Next Sample");
        self.segmentation_box_grid_layout.addWidget(self.next_sample_button, 9, 0, 1, 1);

        self.submit_label_button = QPushButton();
        self.submit_label_button.setText("Submit Label");
        self.segmentation_box_grid_layout.addWidget(self.submit_label_button, 9, 1, 1, 1);
        items_column1+=1;

        self.update_model_button = QPushButton();
        self.update_model_button.setText("Update Model");
        self.segmentation_box_grid_layout.addWidget(self.update_model_button, 10, 0, 1, 1);

        self.predict_button = QPushButton();
        self.predict_button.setText("Predict");
        self.segmentation_box_grid_layout.addWidget(self.predict_button, 10, 1, 1, 1);
        items_column1+=1;

        self.editable_segments_label = QLabel();
        self.editable_segments_label.setText("Labeled radiographs");
        self.segmentation_box_grid_layout.addWidget(self.editable_segments_label, 11, 0, 1, 1);
        items_column1+=1;

        self.editable_segments_list = QListWidget();
        self.segmentation_box_grid_layout.addWidget(self.editable_segments_list, 12, 0, 1, 2);
        items_column1+=1;
        
        self.box_segmentation.setLayout(self.segmentation_box_grid_layout);
        self.gridLayout.addWidget(self.box_segmentation,1,0,items_column1,2);
        

        self.clahe_slider = QSlider(Qt.Orientation.Horizontal);
        self.clahe_slider.setMinimum(0);
        self.clahe_slider.setMaximum(23);
        self.clahe_slider.setValue(0);
        self.box_image_processing_layout.addWidget(self.clahe_slider, items_column1, 0,1,2);
        self.box_image_processing.setLayout(self.box_image_processing_layout);
        items_column1+=1;

        self.clip_limit_slider = QSlider(Qt.Orientation.Horizontal);
        self.clip_limit_slider.setMinimum(0);
        self.clip_limit_slider.setMaximum(23);
        self.clip_limit_slider.setValue(0);
        self.box_image_processing_layout.addWidget(self.clip_limit_slider, items_column1, 0,1,2);
        self.box_image_processing.setLayout(self.box_image_processing_layout);
        items_column1+=1;

        self.gridLayout.addWidget(self.box_image_processing, items_column1, 0, 2, 2);
        #-------------------------------------------------------------
        
        #Column 1
        self.train_combo_box = QComboBox(self);
        self.train_combo_box.addItem("Cuda");
        self.train_combo_box.addItem("Cpu");
        self.box_train_grid.addWidget(self.train_combo_box,1,1,1,1);
        #-------------------------------------------------------------

        #Row contraction
        self.verticalSpacer = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding) 
        self.gridLayout.addItem(self.verticalSpacer, items_column1, 0);
        #--------------------------------------------------------------

        #Graphics views in column 3 and 4
        self.radiograph_view = RadiographViewer(self);
        self.gridLayout.addWidget(self.radiograph_view, 0, 2, items_column1+1, 2)
        

        self.radiograph_label = QLabel();
        self.radiograph_label.setText("Radiograph");
        self.gridLayout.addWidget(self.radiograph_label, items_column1+1,2,1,1);
        #-------------------------------------------------------------

        #Statusbar
        self.status_bar = QStatusBar();
        self.setStatusBar(self.status_bar);

        self.coordinate_label_status_bar = QLabel();
        self.coordinate_label_status_bar.setText("Mouse Position (0,0)");
        self.coordinate_label_status_bar.setStyleSheet("font: 12px")
        self.status_bar.addWidget(self.coordinate_label_status_bar);
        #Set coordinate label in radiograph view
        self.radiograph_view.set_coordinate_label(self.coordinate_label_status_bar);

        self.file_info_label_status_bar = QLabel();
        self.file_info_label_status_bar.setText("File");
        self.status_bar.addWidget(self.file_info_label_status_bar);
        #-------------------------------------------------------------

        #Connect signals to slots
        self.delete_segmentation_button.clicked.connect(self.delete_segmentation);
        self.segments_list.itemClicked.connect(self.list_item_changed);
        self.editable_segments_list.itemDoubleClicked.connect(self.editable_sgement_item_selected);
        self.paint_button.clicked.connect(self.paint_clicked);
        self.erase_button.clicked.connect(self.erase_clicked);
        self.next_sample_button.clicked.connect(self.next_sample_clicked);
        self.submit_label_button.clicked.connect(self.submit_label_clicked);
        self.update_model_button.clicked.connect(self.update_model_clicked);
        self.predict_button.clicked.connect(self.predict_clicked);
        self.size_slider.valueChanged.connect(self.size_value_changed);
        self.opacity_marker_slider.valueChanged.connect(self.marker_opacity_value_changed);
        self.opacity_layer_slider.valueChanged.connect(self.layer_opacity_value_changed);
        self.clahe_slider.valueChanged.connect(self.clahe_value_changed);
        self.clip_limit_slider.valueChanged.connect(self.clip_limit_value_changed);
        self.add_segmentation_button.clicked.connect(self.add_segmentation_clicked);
        self.segmentation_options_window.confirm_clicked_signal.connect(self.add_segmentation_slot);
        self.layer_selection_window.confirm_clicked_signal.connect(self.confirm_labels_clicked);
        #-------------------------------------------------------------

        self.gridLayout.setColumnStretch(0,1);
        self.gridLayout.setColumnStretch(1,1);
        self.gridLayout.setColumnStretch(2,10);

        self.show();

    def get_next_unlabeled(self):
        self.radiograph_view.clear_whole();
        pixmap = self.data_pool_handler.next_unlabeled();
        if pixmap is not None:
            self.radiograph_view.set_image(pixmap);
            #clear all segmentations
            self.segments_list.clear();
    
    def save_project(self, show_dialog = True):
        self.save_project_signal.emit(self.data_pool_handler.data_list, show_dialog);
    
    def predict_on_unlabeled(self):
        if self.radiograph_view.layer_count != 0:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Icon.Warning)
            msgBox.setText("If you continue this operation, all already drawn layers will be deleted and you already have layers that are not submitted. Do you want to continue? If you continue, you won't be able to undo.")
            msgBox.setWindowTitle("Prediction Confirmation")
            msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            #msgBox.buttonClicked.connect(msgButtonClick)

            return_value = msgBox.exec()

            #We first just try to load the mode. If the loading is successful, we start the prediction.
            if return_value == QMessageBox.Yes:
                self.busy_indicator_window.show();
                self.load_model_signal.emit();
        else:
            self.busy_indicator_window.show();
            self.load_model_signal.emit();
    
    def update_file_info_label(self):
        total = len(self.data_pool_handler.data_list);
        unlabeled = len(self.data_pool_handler.get_all_unlabeled());
        self.file_info_label_status_bar.setText(f'Total radiographs: {total}\tTotal labeled: {total - unlabeled}');

    def update_editable_segments_list(self):
        #First clear the list
        self.editable_segments_list.clear();
        #Update the list of available already labeled radiographs
        dl = self.data_pool_handler.data_list;
        for r in dl.keys():
            if dl[r][0] == 'labeled':
                self.editable_segments_list.addItem(r);
    
    def submit_label(self):
        #Aggregate all layers and submit
        #Save each layer with its given name
        layers = [];
        for idx in range(self.radiograph_view.layer_count):
            itm = self.segments_list.item(idx);
            name = itm.text();
            lbl = self.radiograph_view.get_layer(idx);
            layers.append([lbl,name]);

        #Submit all layers and save meta data
        self.data_pool_handler.submit_label(layers);
            
        self.save_project(False);
        self.get_next_unlabeled();
        self.update_file_info_label();
        self.update_editable_segments_list();
    
    def closeEvent(self, event):
        #Save most recent project first.
        self.save_most_recent_project_signal.emit();
        #Ask user to save or not, if yes, save the project otherwise ignore and exit.
        #QMessageBox.Icon.Critical, "No layer found, image should have at least one layer to submit.", "No layer found", QMessageBox.Ok
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Icon.Warning)
        msgBox.setText("Do you want to save?")
        msgBox.setWindowTitle("Save project")
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        #msgBox.buttonClicked.connect(msgButtonClick)

        return_value = msgBox.exec()
        if return_value == QMessageBox.Yes:
            self.save_project(False);

    #Slots
    def list_item_changed(self,item):
        if item.checkState() == QtCore.Qt.Checked:
            item.setIcon(QtGui.QIcon(self.open_eye_icon))
            idx = self.segments_list.row(item);
            self.radiograph_view.set_layer_visibility(idx, True)
        elif item.checkState() == QtCore.Qt.Unchecked:
            idx = self.segments_list.row(item);
            self.radiograph_view.set_layer_visibility(idx, False)
            item.setIcon(QtGui.QIcon(self.close_eye_icon))
            
        idx = self.segments_list.currentRow();
        self.radiograph_view.set_active_layer(idx);

        #set slider opacity value to the opacity of the selected layer.
        self.opacity_layer_slider.setValue(self.radiograph_view.get_active_layer().opacity() * 255);
        
    def size_value_changed(self, val):
        self.size_label.setText(f"Brush size: {val}");
        self.radiograph_view.brush_size = val;
    
    def size_value_changed_slot(self, val):
        self.size_slider.setValue(val);
    
    def layer_opacity_value_changed(self,val):
        self.opacity_layer_label.setText(f"Layer Opacity: {val}");
        self.radiograph_view.set_layer_opacity(val);
    
    def marker_opacity_value_changed(self,val):
        self.opacity_marker_label.setText(f"Marker Opacity: {val}");
        self.radiograph_view.set_mouse_layer_opacity(val);

    def select_folder_slot(self):
        options = QFileDialog.Options()
        dialog = QFileDialog()
        dialog.setOptions(options)

        dialog.setFilter(dialog.filter() | QtCore.QDir.Hidden)

        dialog.setFileMode(QFileDialog.DirectoryOnly)

        dialog.setAcceptMode(QFileDialog.AcceptOpen);
        dialog.setDirectory(QtCore.QDir.currentPath())

        if dialog.exec_() == QDialog.Accepted:
            paths = dialog.selectedFiles()[0];
            self.data_pool_handler.add_from_folder(paths);
    
    def select_files_slot(self):
        options = QFileDialog.Options()
        dialog = QFileDialog()
        dialog.setOptions(options)

        dialog.setFilter(dialog.filter() | QtCore.QDir.Hidden)

        dialog.setFileMode(QFileDialog.ExistingFiles)

        dialog.setAcceptMode(QFileDialog.AcceptOpen);
        dialog.setDirectory(QtCore.QDir.currentPath())

        if dialog.exec_() == QDialog.Accepted:
            path = dialog.selectedFiles();
            self.data_pool_handler.add_from_files(path);

    def add_segmentation_clicked(self):
        layer_names = self.radiograph_view.get_layer_names();
        Config.PROJECT_PREDEFINED_NAMES.clear();
        Config.PROJECT_PREDEFINED_NAMES.append(deepcopy(Config.PREDEFINED_NAMES));
        self.get_project_names_signal.emit();
        self.segmentation_options_window.show_window(layer_names, QColor(0,0,0));

    def add_segmentation_slot(self, name, color):
        if name not in Config.PROJECT_PREDEFINED_NAMES[0]:
            self.update_meta_signal.emit(name);

        item = QListWidgetItem();
        item.setCheckState(QtCore.Qt.Checked)
        item.setIcon(QtGui.QIcon(self.open_eye_icon))
        
        item.setText(name);
        self.segments_list.addItem(item);
        self.segments_list.setCurrentItem(item);
        color = getrgb(color);
        
        self.radiograph_view.add_layer(name, color);
        self.opacity_layer_slider.setValue(255);

    def delete_segmentation(self):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Icon.Warning)
        msgBox.setText("Do you want to delete segmentation? If you delete, you won't be able to undo this operation.")
        msgBox.setWindowTitle("Delete Confirmation")
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        return_value = msgBox.exec()
        if return_value == QMessageBox.Yes:
            
            idx = self.segments_list.currentRow();
            idxs = self.segments_list.selectedIndexes();
            for i in idxs:
                itm = self.segments_list.itemFromIndex(i);
                r = self.segments_list.row(itm);
                self.segments_list.takeItem(r);
                self.radiograph_view.delete_layer(r);
            
            #Always select index 0 after deletation of a layer
            #First check if we have layer left
            if self.radiograph_view.layer_count != 0:
                #self.segments_list.setCurrentRow(0);
                idx = self.segments_list.currentRow();
                self.radiograph_view.set_active_layer(idx);
            #If we don't have any layers left, change mouse color to white
            else:
                self.radiograph_view.m_mouse_layer.pen_color = QtGui.QColor(255,255,255);

    def load_finished(self):
        self.update_file_info_label();

        #Update the list of available already labeled radiographs
        dl = self.data_pool_handler.data_list;
        for r in dl.keys():
            if dl[r][0] == 'labeled':
                self.editable_segments_list.addItem(r);
    
        #Load and set a random image from unlabeled data pool
        pixmap = self.data_pool_handler.load_random_unlabeled();
        if pixmap is not None:
            self.radiograph_view.set_image(pixmap);

    def next_sample_clicked(self):
        if self.radiograph_view.layer_count != 0:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Icon.Warning)
            msgBox.setText("You have layers that haven't been submitted. If you move to next sample, you won't be able to undo this operation. Do you want to submit them now?")
            msgBox.setWindowTitle("Next Sample Confirmation")
            msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            #msgBox.buttonClicked.connect(msgButtonClick)

            return_value = msgBox.exec()
            if return_value == QMessageBox.No:
                #Get next unlabeled data from data pool and show it
                self.get_next_unlabeled();
            if return_value == QMessageBox.Yes:
                self.submit_label();
        else:
            self.get_next_unlabeled();
    
    def submit_label_clicked(self):
        if self.radiograph_view.layer_count != 0:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Icon.Warning)
            msgBox.setText("Do you want to submit the label?")
            msgBox.setWindowTitle("Submit Label Confirmation")
            msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            #msgBox.buttonClicked.connect(msgButtonClick)

            return_value = msgBox.exec()
            if return_value == QMessageBox.Yes:
                self.submit_label();
        else:
            show_dialoge(QMessageBox.Icon.Critical, "No layer found, image should have at least one layer to submit.", "No layers found", QMessageBox.Ok);
        pass
        
    def paint_clicked(self):
        self.paint_button.setStyleSheet("background-color: Aquamarine")
        self.erase_button.setStyleSheet("background-color: white")
        self.radiograph_view.set_state(LayerItem.DrawState);
        pass
    
    def erase_clicked(self):
        self.erase_button.setStyleSheet("background-color: Aquamarine")
        self.paint_button.setStyleSheet("background-color: white")
        self.radiograph_view.set_state(LayerItem.EraseState);
        pass

    def update_model_clicked(self):
        #Remove all files in predictions
        lst = glob('predictions\\*');
        for f in lst:
            os.unlink(f);
        
        #Check all labeled images labels and show dialoge to choose from
        mask_name_list = [];
        layer_count = dict();
        _, mask_list = get_radiograph_label_meta(os.path.sep.join([Config.PROJECT_ROOT,'images']), 
            os.path.sep.join([Config.PROJECT_ROOT,'labels']));
        for i in range(len(mask_list)):
            df = pd.read_pickle(mask_list[i]);
            names = list(df.keys());
            mask_name_list.append(names);
        
        for i in range(len(mask_name_list)):
            for j in range(len(mask_name_list[i])):
                if mask_name_list[i][j] in layer_count:
                    layer_count[mask_name_list[i][j]] += 1;
                else:
                    layer_count[mask_name_list[i][j]] = 1;
        
        self.layer_selection_window.show_window(layer_count);
        
    def confirm_labels_clicked(self, layers_dict):
        self.start_train_signal.emit(layers_dict);
        self.trainig_info_window.show_window();
        #self.th.start();
        pass
    
    def predict_clicked(self):
        self.predict_on_unlabeled();
    
    def model_loaded_finished_slot(self, b):
        #predict on the current unlabeled data and save the prediction in folder
        #always load model at first
        if b:
            lbl = self.data_pool_handler.current_radiograph;
            self.predict_on_unlabeled_signal.emit(lbl, self.data_pool_handler.data_list);
        else:
            self.busy_indicator_window.hide();
            show_dialoge(QMessageBox.Icon.Critical, f"No model found, train first.", "No model found", QMessageBox.Ok);

    def predict_on_unlabeled_finished(self, layers, layer_names):
        #overlay prediciton on current image
        self.radiograph_view.clear_layers();
        self.segments_list.clear();
        # #Add an item to segmentation list for each layer predicted
        for l in range(len(layers)):
            item = QListWidgetItem();
            item.setCheckState(QtCore.Qt.Checked)
            item.setIcon(QtGui.QIcon(self.open_eye_icon))
            item.setText(layer_names[l]);
            self.segments_list.addItem(item);
            self.segments_list.setCurrentItem(item);

            self.radiograph_view.add_layer(layer_names[l], getrgb(Config.PREDEFINED_COLORS[l]));

            height, width, _ = layers[l].shape

            bytesPerLine = 3 * width
            qImg = QImage(layers[l].data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg);

            mask = pixmap.createMaskFromColor(QtGui.QColor(0,0,0),Qt.MaskMode.MaskInColor);
            pixmap.setMask(mask);
            self.radiograph_view.set_layer_pixmap(l,pixmap);
        
        self.busy_indicator_window.hide();

    def update_model_finished_slot(self):
        self.trainig_info_window.hide();
        show_dialoge(QMessageBox.Icon.Information, f"Train finished", "Info", QMessageBox.Ok);
        #self.predict_on_unlabeled();
        pass

    def set_project_name_slot(self, str):
        self.setWindowTitle('Active Learning - ' + str);

    def save_clicked_slot(self):
        self.save_project(True);
    
    def new_project_slot(self):
        name = QFileDialog.getSaveFileName(window, 'Save File');
        window.confirm_new_project_clicked_signal.emit(name[0]);

    def setup_new_project_slot(self):
        #Setup new project, remove every loaded item.
        self.radiograph_view.clear_whole();
        self.data_pool_handler.clear_datalist();
        self.segments_list.clear();
        self.editable_segments_list.clear();
        self.update_file_info_label();
    
    def open_project_slot(self):
        options = QFileDialog.Options()
        dialog = QFileDialog()
        dialog.setOptions(options)

        dialog.setFilter(dialog.filter() | QtCore.QDir.Hidden)

        dialog.setFileMode(QFileDialog.ExistingFiles)

        dialog.setAcceptMode(QFileDialog.AcceptOpen);
        dialog.setDirectory(QtCore.QDir.currentPath())

        if dialog.exec_() == QDialog.Accepted:
            path = dialog.selectedFiles()[0];
            self.open_project_signal.emit(path);
    
    def editable_sgement_item_selected(self, item):
        #We add items in list by their name in disk, so we only read names and load image with labels.
        radiograph_pixmap, mask_pixmap_list = self.data_pool_handler.load_radiograph(item.text());
        self.data_pool_handler.current_radiograph = item.text();
        self.radiograph_view.clear_whole();
        self.segments_list.clear();

        #Load radiograph view with all items
        self.radiograph_view.set_image(radiograph_pixmap);
        for m in range(len(mask_pixmap_list)):
            item = QListWidgetItem();
            item.setCheckState(QtCore.Qt.Checked)
            item.setIcon(QtGui.QIcon(self.open_eye_icon))
            
            item.setText(mask_pixmap_list[m][0]);
            self.segments_list.addItem(item);
            self.segments_list.setCurrentItem(item);
            color = getrgb(mask_pixmap_list[m][1]);
            self.radiograph_view.add_layer(mask_pixmap_list[m][0], color);
            self.radiograph_view.set_layer_pixmap(m, mask_pixmap_list[m][2]);  
        
        self.radiograph_view.set_layer_opacity(self.opacity_layer_slider.value());
    def save_as_slot(self):
        name = QFileDialog.getSaveFileName(self, 'Save File');
        self.save_project_as_signal.emit(name[0]);
    
    def clahe_value_changed(self):
        radiograph_image = cv2.imread(os.path.sep.join([Config.PROJECT_ROOT, 'images', self.data_pool_handler.current_radiograph]),cv2.IMREAD_GRAYSCALE);
        if self.clahe_slider.value() > 0:
            clahe = cv2.createCLAHE(self.clip_limit_slider.value(),(self.clahe_slider.value(),self.clahe_slider.value()));
            radiograph_image = clahe.apply(radiograph_image);
            height, width = radiograph_image.shape

            bytesPerLine = 1 * width
            qImg = QImage(radiograph_image, width, height, bytesPerLine, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qImg);
            self.radiograph_view.set_image(pixmap, reset=False);
        else:
            pixmap = QPixmap(os.path.sep.join([Config.PROJECT_ROOT, 'images', self.data_pool_handler.current_radiograph]));
            self.radiograph_view.set_image(pixmap, reset=False);
    
    def clip_limit_value_changed(self):
        radiograph_image = cv2.imread(os.path.sep.join([Config.PROJECT_ROOT, 'images', self.data_pool_handler.current_radiograph]),cv2.IMREAD_GRAYSCALE);
        if self.clip_limit_slider.value() > 0 and self.clahe_slider.value() > 0:
            clahe = cv2.createCLAHE(self.clip_limit_slider.value(),(self.clahe_slider.value(),self.clahe_slider.value()));
            radiograph_image = clahe.apply(radiograph_image);
            height, width = radiograph_image.shape

            bytesPerLine = 1 * width
            qImg = QImage(radiograph_image, width, height, bytesPerLine, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qImg);
            self.radiograph_view.set_image(pixmap, reset=False);
        else:
            pixmap = QPixmap(os.path.sep.join([Config.PROJECT_ROOT, 'images', self.data_pool_handler.current_radiograph]));
            self.radiograph_view.set_image(pixmap, reset=False);

    #--------------------------------------------------------------
    
if __name__=='__main__':
   # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    app = QApplication(sys.argv);
    app.setStyleSheet("QPushButton {\
    background-color : white;\
    border-style: outset;\
    border-width: 1px;\
    border-radius: 10px;\
    border-color: beige;\
    font: 12px;\
    min-width: 9em;\
    padding: 8px;\
    }\
    QListView {\
    background-color : HoneyDew;\
    font: 15px;\
    }\
    QPushButton:pressed { background-color: azure; font: 14px;}");


    #Create singletons
    data_pool_handler = DataPoolHandler();
    network_trainer = NetworkTrainer();
    window = MainWindow(data_pool_handler = data_pool_handler, network_trainer = network_trainer);
    project_handler = ProjectHandler();
    new_project_window = NewProjectWindow();
    #---------------------------------------------------------

    # logging.basicConfig(filename='log.txt', level=logging.CRITICAL);

    # root = logging.getLogger()
    # root.setLevel(logging.CRITICAL)

    # handler = logging.StreamHandler(sys.stdout)
    # handler.setLevel(logging.CRITICAL)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # handler.setFormatter(formatter)
    # root.addHandler(handler)
    
    logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
            filename='out.log',
            filemode='a'
            )

    def my_handler(type, value, tb):
        for line in traceback.TracebackException(type, value, tb).format(chain=True):
            logging.exception(line)
        logging.exception(value)

        sys.__excepthook__(type, value, tb) # calls default excepthook
    console = logging.StreamHandler(sys.stdout);
    console.setLevel(logging.ERROR);
    logging.getLogger('').addHandler(console);
    logging.info("info from logging");

    #log = logging.getLogger('exception')
    sys.excepthook = my_handler
    # sys.stdout = LoggerWriter(log.debug)
    # sys.stderr = LoggerWriter(log.error)
    #print('Test to standard out')
    #raise Exception('Test to standard error')
    #sys.stdout.close()

    data_pool_handler.load_finished_signal.connect(window.load_finished);
    data_pool_handler.save_project_signal.connect(window.save_project);
    window.start_train_signal.connect(network_trainer.start_train_slot);
    network_trainer.train_finsihed_signal.connect(window.update_model_finished_slot);
    window.save_project_signal.connect(project_handler.save_project);
    project_handler.open_project_signal.connect(data_pool_handler.open_project_slot);
    window.load_model_signal.connect(network_trainer.load_model);
    window.predict_on_unlabeled_signal.connect(network_trainer.predict);
    network_trainer.predict_finished_signal.connect(window.predict_on_unlabeled_finished);
    project_handler.set_project_name_signal.connect(window.set_project_name_slot);
    project_handler.new_project_setup_signal.connect(window.setup_new_project_slot);
    window.open_project_signal.connect(project_handler.open_project);
    new_project_window.open_project_signal.connect(project_handler.open_project);
    window.save_most_recent_project_signal.connect(project_handler.save_most_recent_project);
    window.save_project_as_signal.connect(project_handler.save_project_as);
    window.radiograph_view.size_changed_signal.connect(window.size_value_changed_slot);
    window.update_meta_signal.connect(project_handler.update_project_meta_slot);
    window.get_project_names_signal.connect(project_handler.get_project_names_slot);
    new_project_window.confirm_new_project_clicked_signal.connect(project_handler.new_project);
    window.confirm_new_project_clicked_signal.connect(project_handler.new_project);
    network_trainer.update_train_info_iter.connect(window.trainig_info_window.update_train_info_iter_slot);
    network_trainer.update_valid_info_iter.connect(window.trainig_info_window.update_valid_info_iter_slot);
    network_trainer.update_train_info_epoch_train.connect(window.trainig_info_window.update_train_info_epoch_train_slot);
    network_trainer.update_train_info_epoch_valid.connect(window.trainig_info_window.update_train_info_epoch_valid_slot);
    network_trainer.augmentation_finished_signal.connect(window.trainig_info_window.augmentation_finished_slot);
    window.trainig_info_window.terminate_signal.connect(network_trainer.terminate_slot);
    network_trainer.model_loaded_finished.connect(window.model_loaded_finished_slot);

    ret = project_handler.open_project();
    
    if not ret:
        #No project found so show project name dialoge.
       
        new_project_window.show_window();
        pass

    sys.exit(app.exec_());