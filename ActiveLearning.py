#==================================================================
#==================================================================
from ast import List
from copy import deepcopy, copy
from posixpath import basename
import logging
import traceback
from re import I
from PIL import ImageColor
from PyQt5.QtCore import QThread, Qt
from PyQt5.QtWidgets import QSizePolicy, QApplication, QCheckBox, QColorDialog, QComboBox, QDesktopWidget, QGridLayout, QGroupBox, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QProgressBar, QPushButton, QRadioButton, QScrollArea, QSlider, QFileDialog, QDialog, QStatusBar, QTabBar, QTabWidget, QTextEdit, QVBoxLayout
import sys
from PyQt5 import QtGui
from PyQt5 import QtWidgets
import Config
import Class
from RadiographViewer import *
from DataPoolHandler import *
from Utility import *
from NetworkTrainer import *
from ProjectHandler import *
from PIL.ImageColor import *
from CustomWidgets import *
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
        self.setGeometry(int(centerPoint.x() - self.width/2), int(centerPoint.y()-self.height/2), self.width, self.height);
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
        self.setGeometry(int(centerPoint.x() - self.width/2), int(centerPoint.y() - self.height/2), self.width, self.height);
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
                self.name_selected = rbtn.text();
            rbtn.toggled.connect(lambda:self.btnstate(rbtn))
            self.box_gridLayout.addWidget(rbtn, num_items, 0, 1, 1);
            num_items+=1;

        self.box_gridLayout.addWidget(self.custom_name,num_items,0,1,1);
        self.custom_name.setChecked(False);

        self.box_gridLayout.addWidget(self.segmentation_name,num_items,1,1,1);
        num_items+=1;

        self.box_gridLayout.addWidget(self.pen_label,num_items,0,1,1);

        self.box_gridLayout.addWidget(self.pen_button,num_items,1,1,1);
        num_items+=1;

        self.box_gridLayout.addWidget(self.confirm_button,num_items,0,1,2);

        self.name_gorupbox.setLayout(self.box_gridLayout);
        self.window_grid_layout.addWidget(self.name_gorupbox,0,0,num_items,1);

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
    
    def check_submit(self):
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
    
    def confirm_clicked(self):
        if self.custom_name.isChecked() == True:
            if self.segmentation_name.text() != '':
                self.name_selected = self.segmentation_name.text();
            else:
                show_dialoge(QMessageBox.Icon.Critical, "Label name cannot be empty", "Empty label name", QMessageBox.Ok);
        
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
        self.layers_dict = deepcopy(layers);
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
            if Config.MUTUAL_EXCLUSION is False:
                Config.NUM_CLASSES = num_selected;
            else:
                Config.NUM_CLASSES = num_selected+1;
            
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
    save_project_signal = pyqtSignal(bool);
    save_project_as_signal = pyqtSignal(str);
    load_model_signal = pyqtSignal();
    predict_on_unlabeled_signal = pyqtSignal(str, dict);
    open_project_signal = pyqtSignal(str);
    save_most_recent_project_signal = pyqtSignal();
    update_meta_signal = pyqtSignal(str);
    get_project_names_signal = pyqtSignal();
    confirm_new_project_clicked_signal = pyqtSignal(str)
    foreground_clicked_signal = pyqtSignal();
    background_clicked_signal = pyqtSignal();
    update_gc_signal = pyqtSignal();
    reset_gc_signal = pyqtSignal();
    update_foreground_with_layer_signal = pyqtSignal();
    add_from_files_signal = pyqtSignal(list);
    add_from_folder_signal= pyqtSignal(str);
    add_from_dicom_folder_signal= pyqtSignal(str);

    def __init__(self):
        super().__init__();
        self.setWindowModality(QtCore.Qt.ApplicationModal)

        #Parameters
        self.title = "Active Learning";
        self.left = 0;
        self.top = 0;
        self.width = 1500;
        self.height = 800;
        self.close_eye_icon = "Icons/eye_icon_closed.png";
        self.open_eye_icon = "Icons/eye_icon.png"
        self.folder_icon = "Icons/folder_icon.png";
        self.dicom_folder_icon = "Icons/dicom_folder_icon.png";
        self.files_icon = "Icons/files_icon.png";
        self.draw_icon = "Icons/draw_icon.png";
        self.erase_icon = "Icons/erase_icon.png";
        self.fill_icon = "Icons/fill-icon.png";
        self.save_icon = "Icons/save_icon.ico";
        self.save_as_icon = "Icons/save_as_icon.png";
        self.open_project_icon = "Icons/open_project_icon.ico";
        self.new_project_icon = "Icons/new_project_icon.png";
        self.undo_icon = "Icons/Undo_Icon.png";
        self.redo_icon = "Icons/Redo_Icon.png";
        self.scissor_icon = "Icons/Scissor_Icon.png";
        self.segmentation_options_window = SegmentationOptionsWindow();
        self.layer_selection_window = LayerSelectionWindow();
        self.trainig_info_window = TrainingInfoWindow();
        self.busy_indicator_window = WaitingWindow();
        self.background_gc_selected = False;
        self.foreground_gc_selected = True;
        self.erase_gc_selected = False;
        #---------------------------------------------------------------

        self.network_trainer_thread = QThread();
        Class.network_trainer.moveToThread(self.network_trainer_thread);
        self.network_trainer_thread.start();

        self.data_pool_handler_thread = QThread();
        Class.data_pool_handler.moveToThread(self.data_pool_handler_thread);
        self.data_pool_handler_thread.start();

        self.init_ui();

    def init_ui(self):
        #Find best location
        centerPoint = QDesktopWidget().availableGeometry();
        pos_x = int((centerPoint.width() - self.width) / 2);
        pos_y = int((centerPoint.height() - self.height) / 2);
        
        self.setGeometry(pos_x, pos_y, self.width, self.height);
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
        self.file_menu.addAction("Open folder",self.open_folder_slot);
        self.file_menu.addAction("Open DICOM folder",self.open_dicom_folder_slot);
        self.file_menu.addAction("Open files",self.select_files_slot);
        #-------------------------------------------------

        #Toolbar
        self.toolbar = self.addToolBar('top_toolbar');
        new_project_action = self.toolbar.addAction('New project', self.new_project_slot);
        new_project_action.setIcon(QtGui.QIcon(self.new_project_icon));
        open_project_action = self.toolbar.addAction('Open project', self.open_project_slot);
        open_project_action.setIcon(QtGui.QIcon(self.open_project_icon));
        select_folder_action = self.toolbar.addAction("Open folder",self.open_folder_slot);
        select_folder_action.setIcon(QtGui.QIcon(self.folder_icon));
        select_dicom_folder_action = self.toolbar.addAction("Open DICOM folder",self.open_dicom_folder_slot);
        select_dicom_folder_action.setIcon(QtGui.QIcon(self.dicom_folder_icon));
        select_files_action = self.toolbar.addAction("Open files",self.select_files_slot);
        select_files_action.setIcon(QtGui.QIcon(self.files_icon));
        save_action = self.toolbar.addAction("Save",self.save_clicked_slot);
        save_action.setIcon(QtGui.QIcon(self.save_icon));
        save_as_action = self.toolbar.addAction("Save As",self.save_as_slot);
        save_as_action.setIcon(QtGui.QIcon(self.save_as_icon));
        undo_action = self.toolbar.addAction("Undo",self.undo_slot);
        undo_action.setIcon(QtGui.QIcon(self.undo_icon));
        redo_action = self.toolbar.addAction("Redo",self.redo_slot);
        redo_action.setIcon(QtGui.QIcon(self.redo_icon));
        #-------------------------------------------------
        
        menu_content = QtWidgets.QWidget();
        menu_panel_scroll_area = QScrollArea();
        menu_panel_scroll_area.setWidgetResizable(True);
        menu_panel_layout = QVBoxLayout(menu_content);

        self.box_layers = CollapsibleBox("Layers");
        self.layers_box_grid_layout = QGridLayout();

        self.box_segmentaion = CollapsibleBox("Segmentation");

        self.box_manual_segmentation = QWidget();
        self.manual_segmentation_grid_layout = QGridLayout();

        self.box_automatic_segmentation = QWidget();
        self.automatic_segmentation_grid_layout = QGridLayout();

        self.box_radiograph_manipulation = CollapsibleBox("Manipulation");
        self.radiograph_manipulation_box_grid_layout = QGridLayout();

        self.box_layers_control = CollapsibleBox("Layer control");
        self.layers_control_grid_layout = QGridLayout();

        self.box_image_processing = CollapsibleBox("Image processing");
        #self.box_image_processing.setTitle();
        self.box_image_processing_layout = QGridLayout();

        self.box_quality_labels_params = CollapsibleBox("Qaulity labels");
        self.box_quality_labels_grid = QGridLayout();
        
        self.segmentation_box_tab = QTabWidget();
        stylesheet = """ 
        QTabBar::tab {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #E1E1E1, stop: 0.4 #DDDDDD,
                                stop: 0.5 #D8D8D8, stop: 1.0 #D3D3D3);
            border: 2px solid #C4C4C3;
            border-bottom-color: #C2C7CB; /* same as the pane color */
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            min-width: 8ex;
            padding: 2px;
            }
        QTabBar::tab:selected, QTabBar::tab:hover {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #fafafa, stop: 0.4 #f4f4f4,
                                stop: 0.5 #e7e7e7, stop: 1.0 #fafafa);
        }


            """
        self.segmentation_box_tab.setStyleSheet(stylesheet)
        
        #Quality
        items_count = 0;   
        
        self.rotation_label = QLabel(self);
        self.rotation_label.setText("Rotation");
        self.box_quality_labels_grid.addWidget(self.rotation_label,0,0,1,1);
        items_count+=1;

        self.exposure_label = QLabel(self);
        self.exposure_label.setText("Exposure");
        self.box_quality_labels_grid.addWidget(self.exposure_label,1,0,1,1);
        items_count+=1;

        self.rotation_combo_box = QComboBox(self);
        self.rotation_combo_box.addItem("Mild");
        self.rotation_combo_box.addItem("Moderate");
        self.rotation_combo_box.addItem("Marked");
        self.rotation_combo_box.addItem("Normal");
        self.box_quality_labels_grid.addWidget(self.rotation_combo_box,0,1,1,1);

        self.exposure_combo_box = QComboBox(self);
        self.exposure_combo_box.addItem("Normal");
        self.exposure_combo_box.addItem("Underexposed-mild");
        self.exposure_combo_box.addItem("Underexposed-moderate");
        self.exposure_combo_box.addItem("Underexposed-marked");
        self.exposure_combo_box.addItem("Overexposed-mild");
        self.exposure_combo_box.addItem("Overexposed-moderate");
        self.exposure_combo_box.addItem("Overexposed-marked");
        self.box_quality_labels_grid.addWidget(self.exposure_combo_box, 1, 1, 1, 1);

        self.box_quality_labels_params.setContentLayout(self.box_quality_labels_grid);
        menu_panel_layout.addWidget(self.box_quality_labels_params);
        #-------------------------------------------------------------------------

        #Layers box
        next_start = items_count;
        items_count = 0;
        self.add_segmentation_button = QPushButton(self);
        self.add_segmentation_button.setText("Add");

        self.layers_box_grid_layout.addWidget(self.add_segmentation_button, items_count, 0,1,2);

        self.delete_segmentation_button = QPushButton(self);
        self.delete_segmentation_button.setText("Delete");
        self.layers_box_grid_layout.addWidget(self.delete_segmentation_button, items_count, 4, 1,2);
        items_count+=1;

        self.segments_list = QListWidget(self);
        self.layers_box_grid_layout.addWidget(self.segments_list,items_count,0,1,6);
        items_count+=1;

        self.box_layers.setContentLayout(self.layers_box_grid_layout);
        menu_panel_layout.addWidget(self.box_layers);
        #-------------------------------------------------------------------

        dummy_layout = QVBoxLayout();
        #Manual segmentation items
        items_count = 0;
        self.paint_button = QPushButton();
        self.paint_button.setText("Paint");
        self.paint_button.setStyleSheet("background-color: Aquamarine")
        self.paint_button.setIcon(QtGui.QIcon(self.draw_icon))
        self.manual_segmentation_grid_layout.addWidget(self.paint_button, items_count, 0, 1, 1);

        self.erase_button = QPushButton();
        self.erase_button.setText("Erase");
        self.erase_button.setStyleSheet("background-color: white")
        self.erase_button.setIcon(QtGui.QIcon(self.erase_icon))
        self.manual_segmentation_grid_layout.addWidget(self.erase_button, items_count, 1, 1, 1);
        items_count+=1;

        self.fill_button = QPushButton();
        self.fill_button.setText("Fill");
        self.fill_button.setStyleSheet("background-color: white")
        self.fill_button.setIcon(QtGui.QIcon(self.fill_icon))
        self.manual_segmentation_grid_layout.addWidget(self.fill_button, items_count, 0, 1, 1);

        self.magnetic_scissor_button = QPushButton();
        self.magnetic_scissor_button.setText("Magnetic Scissor");
        self.magnetic_scissor_button.setStyleSheet("background-color: white")
        self.magnetic_scissor_button.setIcon(QtGui.QIcon(self.scissor_icon))
        self.manual_segmentation_grid_layout.addWidget(self.magnetic_scissor_button, items_count, 1, 1, 1);
        items_count+=1;

        self.box_manual_segmentation.setLayout(self.manual_segmentation_grid_layout);
        self.segmentation_box_tab.addTab(self.box_manual_segmentation, 'Manual segmentation')

        #------------------------------------------------------------------

        #automatic segmentation items
        items_count = 0;
        self.foreground_button = QPushButton();
        self.foreground_button.setText("Foreground");
        self.foreground_button.clicked.connect(self.foreground_clicked);
        self.foreground_button.setStyleSheet("background-color: white");
        self.automatic_segmentation_grid_layout.addWidget(self.foreground_button, items_count, 0, 1, 1);

        self.background_button = QPushButton();
        self.background_button.setText("Background");
        self.background_button.setStyleSheet("background-color: white")
        self.background_button.clicked.connect(self.background_clicked);
        self.automatic_segmentation_grid_layout.addWidget(self.background_button, items_count, 1, 1, 1);
        items_count+=1;

        self.erase_gc_button = QPushButton();
        self.erase_gc_button.setText("Erase");
        self.erase_gc_button.setStyleSheet("background-color: white")
        self.erase_gc_button.setIcon(QtGui.QIcon(self.erase_icon))
        self.automatic_segmentation_grid_layout.addWidget(self.erase_gc_button, items_count, 0, 1, 1);

        self.update_gc_button = QPushButton();
        self.update_gc_button.setText("Update segmentation");
        self.update_gc_button.clicked.connect(self.update_gc_clicked);
        self.automatic_segmentation_grid_layout.addWidget(self.update_gc_button, items_count, 1, 1, 1);
        items_count+=1;

        self.reset_gc = QPushButton();
        self.reset_gc.setText("Reset");
        self.reset_gc.clicked.connect(self.reset_gc_clicked);
        self.automatic_segmentation_grid_layout.addWidget(self.reset_gc, items_count, 0, 1, 1);

        self.update_foreground_with_layer_button = QPushButton();
        self.update_foreground_with_layer_button.setText("Update foreground");
        self.update_foreground_with_layer_button.setWhatsThis("Update foreground layer with what you've already drawn on the radiograph. This will automatically remove anything you already drew as the foreground")
        self.update_foreground_with_layer_button.clicked.connect(self.update_foreground_with_layer_clicked);
        self.automatic_segmentation_grid_layout.addWidget(self.update_foreground_with_layer_button, items_count, 1, 1, 1);

        self.foreground_button.setStyleSheet("background-color: Aquamarine")
        self.background_button.setStyleSheet("background-color: White")
        self.erase_gc_button.setStyleSheet("background-color: White")

        self.box_automatic_segmentation.setLayout(self.automatic_segmentation_grid_layout);
        self.segmentation_box_tab.addTab(self.box_automatic_segmentation, 'Automatic segmentation')
        #self.scroll_area_automatic_segmentation.setWidget(self.box_automatic_segmentation);
        #self.scroll_area_automatic_segmentation.setFixedHeight(250);

        #------------------------------------------------------------------
        dummy_layout.addWidget(self.segmentation_box_tab);
        self.box_segmentaion.setContentLayout(dummy_layout);
        menu_panel_layout.addWidget(self.box_segmentaion);

        #layers control box
        items_count = 0;
        self.size_label = QLabel(self);
        self.size_label.setText("Brush size: 150");
        self.layers_control_grid_layout.addWidget(self.size_label, items_count, 0, 1, 1);
        items_count+=1;

        self.size_slider = QSlider(Qt.Orientation.Horizontal);
        self.size_slider.setMinimum(1);
        self.size_slider.setMaximum(Config.MAX_PEN_SIZE);
        self.size_slider.setValue(150);
        self.layers_control_grid_layout.addWidget(self.size_slider, items_count,0,1,2);
        items_count+=1;

        self.opacity_layer_label = QLabel(self);
        self.opacity_layer_label.setText("Layer Opacity: 255");
        self.layers_control_grid_layout.addWidget(self.opacity_layer_label, items_count, 0, 1, 1);
        items_count+=1;

        self.opacity_layer_slider = QSlider(Qt.Orientation.Horizontal);
        self.opacity_layer_slider.setMinimum(0);
        self.opacity_layer_slider.setMaximum(255);
        self.opacity_layer_slider.setValue(255);
        self.layers_control_grid_layout.addWidget(self.opacity_layer_slider, items_count,0,1,2);
        items_count+=1;

        self.opacity_marker_label = QLabel(self);
        self.opacity_marker_label.setText("Marker Opacity: 255");
        self.layers_control_grid_layout.addWidget(self.opacity_marker_label, items_count, 0, 1, 1);
        items_count+=1;

        self.opacity_marker_slider = QSlider(Qt.Orientation.Horizontal);
        self.opacity_marker_slider.setMinimum(0);
        self.opacity_marker_slider.setMaximum(255);
        self.opacity_marker_slider.setValue(255);
        self.layers_control_grid_layout.addWidget(self.opacity_marker_slider, items_count,0,1,2);
        items_count+=1;

        self.box_layers_control.setContentLayout(self.layers_control_grid_layout);
        menu_panel_layout.addWidget(self.box_layers_control);
        #-----------------------------------------------------------

        #Box radiograph manipulation
        #next_start = self.gridLayout.rowCount();
        items_count = 0;
        self.next_sample_button = QPushButton();
        self.next_sample_button.setText("Next Sample");
        self.radiograph_manipulation_box_grid_layout.addWidget(self.next_sample_button, items_count, 0, 1, 1);

        self.submit_label_button = QPushButton();
        self.submit_label_button.setText("Submit Label");
        self.radiograph_manipulation_box_grid_layout.addWidget(self.submit_label_button, items_count, 1, 1, 1);
        items_count+=1;

        self.update_model_button = QPushButton();
        self.update_model_button.setText("Update Model");
        self.radiograph_manipulation_box_grid_layout.addWidget(self.update_model_button, items_count, 0, 1, 1);

        self.predict_button = QPushButton();
        self.predict_button.setText("Predict");
        self.radiograph_manipulation_box_grid_layout.addWidget(self.predict_button, items_count, 1, 1, 1);
        items_count+=1;

        self.radiographs_list_label = QLabel();
        self.radiographs_list_label.setText("All radiographs");
        self.radiograph_manipulation_box_grid_layout.addWidget(self.radiographs_list_label, items_count, 0, 1, 1);
        items_count+=1;

        self.all_radiographs_list = QListWidget();
        self.radiograph_manipulation_box_grid_layout.addWidget(self.all_radiographs_list, items_count, 0, 1, 2);
        self.all_radiographs_list.setFixedHeight(250)
        items_count+=1;
        
        self.box_radiograph_manipulation.setContentLayout(self.radiograph_manipulation_box_grid_layout);
        menu_panel_layout.addWidget(self.box_radiograph_manipulation);
        #----------------------------------------------------------------

        #Image processing
        #next_start += items_count;
        items_count = 0;
        self.clahe_slider = QSlider(Qt.Orientation.Horizontal);
        self.clahe_slider.setMinimum(0);
        self.clahe_slider.setMaximum(50);
        self.clahe_slider.setValue(8);
        self.box_image_processing_layout.addWidget(self.clahe_slider, items_count, 0,1,2);
        items_count+=1;

        self.clip_limit_slider = QSlider(Qt.Orientation.Horizontal);
        self.clip_limit_slider.setMinimum(0);
        self.clip_limit_slider.setMaximum(50);
        self.clip_limit_slider.setValue(2);
        self.box_image_processing_layout.addWidget(self.clip_limit_slider, items_count, 0,1,2);
        items_count+=1;

        self.box_image_processing.setContentLayout(self.box_image_processing_layout);
        menu_panel_layout.addWidget(self.box_image_processing,);
        #-------------------------------------------------------------

        menu_panel_layout.addStretch();
        menu_panel_scroll_area.setWidget(menu_content);

        self.gridLayout.addWidget(menu_panel_scroll_area,0,0,1,6);

        # #Row contraction
        # self.verticalSpacer = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding) 
        # self.gridLayout.addItem(self.verticalSpacer, items_count, 0);
        # #--------------------------------------------------------------

        # #Graphics views
        self.radiograph_view = RadiographViewer(self);
        self.gridLayout.addWidget(self.radiograph_view, 0, 6, self.gridLayout.rowCount(), 2)
        

        self.radiograph_label = QLabel();
        self.radiograph_label.setText("Radiograph");
        self.gridLayout.addWidget(self.radiograph_label, self.gridLayout.rowCount()+1,6,1,2);
        # #-------------------------------------------------------------

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
        self.paint_button.clicked.connect(self.paint_clicked);
        self.erase_button.clicked.connect(self.erase_clicked);
        self.erase_gc_button.clicked.connect(self.erase_gc_clicked);
        self.fill_button.clicked.connect(self.fill_clicked);
        self.magnetic_scissor_button.clicked.connect(self.magnetic_scissor_clicked);
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
        self.segmentation_box_tab.currentChanged.connect(self.segmentation_tab_changed);
        #-------------------------------------------------------------

        for i in range(6):
            self.gridLayout.setColumnStretch(i,1);
        self.gridLayout.setColumnStretch(6,20);

        self.show();

    def get_next_unlabeled(self):
        self.radiograph_view.clear_whole();
        pixmap = Class.data_pool_handler.next_unlabeled();
        #Update name
        self.radiograph_label.setText(f"Radiograph Name: {Class.data_pool_handler.current_radiograph}")
        if pixmap is not None:
            self.radiograph_view.set_image(pixmap);
            #clear all segmentations
            self.segments_list.clear();
    
    def save_project(self, show_dialog = True):
        self.save_project_signal.emit(show_dialog);
    
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
        total = len(Class.data_pool_handler.data_list);
        unlabeled = len(Class.data_pool_handler.get_all());
        self.file_info_label_status_bar.setText(f'Total radiographs: {total}\tTotal labeled: {total - unlabeled}');

    def update_all_radiographs_segments_list(self):
        #First clear the list
        self.all_radiographs_list.clear();
        #Update the list of available already labeled radiographs
        dl = Class.data_pool_handler.data_list;
        for r in dl.keys():
            if dl[r][0] == 'labeled':
                list_item_meta = LabelledRadListItem();
                list_item_meta.set_name(r, '(0,0,255)')
                list_item_meta.set_status('Labeled' ,'(0,255,0)')
                list_item_meta.open_radiograph_signal.connect(self.load_radiograph_slot);
                list_item_meta.delete_radiograph_signal.connect(self.delete_radiograph_slot);
                list_widget_item = QListWidgetItem(self.all_radiographs_list);
                list_widget_item.setSizeHint(list_item_meta.sizeHint())
                self.all_radiographs_list.addItem(list_widget_item);
                self.all_radiographs_list.setItemWidget(list_widget_item, list_item_meta);
            elif dl[r][0] == 'unlabeled':
                list_item_meta = LabelledRadListItem();
                list_item_meta.set_name(r, '(0,0,255)')
                list_item_meta.set_status('Unlabeled','(255,0,0)');
                list_item_meta.open_radiograph_signal.connect(self.load_radiograph_slot);
                list_item_meta.delete_radiograph_signal.connect(self.delete_radiograph_slot);
                list_widget_item = QListWidgetItem(self.all_radiographs_list);
                list_widget_item.setSizeHint(list_item_meta.sizeHint())
                self.all_radiographs_list.addItem(list_widget_item);
                self.all_radiographs_list.setItemWidget(list_widget_item, list_item_meta);
    
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
        Class.data_pool_handler.submit_label(layers, self.rotation_combo_box.currentText(), self.exposure_combo_box.currentText());
            
        self.save_project(False);
        self.get_next_unlabeled();
        self.update_file_info_label();
        self.update_all_radiographs_segments_list();
    
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
    
    def automatic_segmentation_enable(self):
        for i in range(self.segments_list.count()):
            itm = self.segments_list.item(i);
            if itm.checkState() == QtCore.Qt.Checked:
                self.radiograph_view.set_gc_layers_visibility(True, i);
            else:
                self.radiograph_view.set_gc_layers_visibility(False, i);

        #Disable regular layer as we don't want it to be affected.
        #self.radiograph_view.set_layer_active(False);
        #self.radiograph_view.set_layer_visibility(False);

        self.radiograph_view.deactive_all_layers();
        self.radiograph_view.hide_all_layers();

        if self.background_gc_selected:
            self.radiograph_view.set_gc_layer_active(True, type=1);
        elif self.foreground_gc_selected:
            self.radiograph_view.set_gc_layer_active(True, type=0);
        
        if self.erase_gc_selected is True:
            self.radiograph_view.set_state(LayerItem.EraseState);
        
        #Check if satate is erase from manual segmentation, then activate
        #erase grabcut button
        if self.radiograph_view.get_state() == LayerItem.EraseState:
            self.paint_button.setStyleSheet("background-color: white")
            self.erase_gc_button.setStyleSheet("background-color: Aquamarine")
            self.fill_button.setStyleSheet("background-color: white")
    
    def manual_segmentation_enable(self):
        #Enable regular layer as we don't want it to be affected.
        for i in range(self.radiograph_view.layer_count):
            self.radiograph_view.set_gc_layer_active(False,i);
            self.radiograph_view.set_gc_layers_visibility(False,i);

        self.radiograph_view.set_layer_active(True);
        
        for i in range(self.segments_list.count()):
            itm = self.segments_list.item(i);
            if itm.checkState() == QtCore.Qt.Checked:
                self.radiograph_view.set_layer_visibility(True, i);
            else:
                self.radiograph_view.set_layer_visibility(False, i);

    
        #By default make paint enabled
        self.paint_button.setStyleSheet("background-color: Aquamarine")
        self.erase_button.setStyleSheet("background-color: white")
        self.fill_button.setStyleSheet("background-color: white")
        self.magnetic_scissor_button.setStyleSheet("background-color: white")
        self.radiograph_view.set_state(LayerItem.DrawState);
    
    def load_radiograph_slot(self, txt):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Icon.Warning)
        msgBox.setText("If you open this sample, you won't be able to undo this operation. Do you want to continue?")
        msgBox.setWindowTitle("Next Sample Confirmation")
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        return_value = msgBox.exec();
        if return_value == QMessageBox.Yes:
            radiograph_pixmap, mask_pixmap_list = Class.data_pool_handler.load_radiograph(txt);
            Class.data_pool_handler.current_radiograph = txt;
            self.radiograph_label.setText(f"Radiograph Name: {Class.data_pool_handler.current_radiograph}")
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
    
    def delete_radiograph_slot(self,txt):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Icon.Warning)
        msgBox.setText("Are you sure? you won't be able to undo this operation.")
        msgBox.setWindowTitle("Delete confirmation")
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        return_value = msgBox.exec()

        #We first just try to load the mode. If the loading is successful, we start the prediction.
        if return_value == QMessageBox.Yes:
            Class.data_pool_handler.delete_radiograph(txt);
            self.update_all_radiographs_segments_list();
            self.update_file_info_label();
            #Update current radiograph if we deleted current one
            if Class.data_pool_handler.current_radiograph == txt:
                self.get_next_unlabeled();
    #Slots
    def list_item_changed(self,item):
        if item.checkState() == QtCore.Qt.Checked:
            item.setIcon(QtGui.QIcon(self.open_eye_icon))
            idx = self.segments_list.row(item);

            if self.segmentation_box_tab.currentIndex() == 0:
                self.radiograph_view.set_layer_visibility(True, idx)
            if self.segmentation_box_tab.currentIndex() == 1:
                self.radiograph_view.set_gc_layers_visibility(True, idx);
                #self.radiograph_view.set_gc_layers_visibility(True, idx);


        elif item.checkState() == QtCore.Qt.Unchecked:
            idx = self.segments_list.row(item);

            if self.segmentation_box_tab.currentIndex() == 0:
                self.radiograph_view.set_layer_visibility(False, idx)
            if self.segmentation_box_tab.currentIndex() == 1:
                self.radiograph_view.set_gc_layers_visibility(False, idx);
                
            item.setIcon(QtGui.QIcon(self.close_eye_icon))
        
        idx = self.segments_list.currentRow();
        #deactive current selected layer
        if self.segmentation_box_tab.currentIndex() == 0:
            self.radiograph_view.set_layer_active(False, idx = self.radiograph_view.get_active_layer_index());
            self.radiograph_view.set_active_layer_index(idx);
            self.radiograph_view.set_layer_active(True, idx);
        elif self.segmentation_box_tab.currentIndex() == 1:
            if self.background_gc_selected is True:
                self.radiograph_view.set_gc_layer_active(False, self.radiograph_view.get_active_layer_index());
                self.radiograph_view.set_active_layer_index(idx);
                self.radiograph_view.set_gc_layer_active(True, idx, 1);
            else:
                self.radiograph_view.set_gc_layer_active(False , self.radiograph_view.get_active_layer_index());
                self.radiograph_view.set_active_layer_index(idx);
                self.radiograph_view.set_gc_layer_active(True, idx, 0);

        #set slider opacity value to the opacity of the selected layer.
        self.opacity_layer_slider.setValue(int(self.radiograph_view.get_active_layer().opacity() * 255));
        
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

    def open_folder_slot(self):
        options = QFileDialog.Options()
        dialog = QFileDialog()
        dialog.setOptions(options)

        dialog.setFilter(dialog.filter() | QtCore.QDir.Hidden)

        dialog.setFileMode(QFileDialog.DirectoryOnly)

        dialog.setAcceptMode(QFileDialog.AcceptOpen);
        dialog.setDirectory(QtCore.QDir.currentPath())

        if dialog.exec_() == QDialog.Accepted:
            self.busy_indicator_window.show();
            path = dialog.selectedFiles()[0];
            self.add_from_folder_signal.emit(path);
    
    def open_dicom_folder_slot(self):
        options = QFileDialog.Options()
        dialog = QFileDialog()
        dialog.setOptions(options)

        dialog.setFilter(dialog.filter() | QtCore.QDir.Hidden)

        dialog.setFileMode(QFileDialog.DirectoryOnly)

        dialog.setAcceptMode(QFileDialog.AcceptOpen);
        dialog.setDirectory(QtCore.QDir.currentPath())

        if dialog.exec_() == QDialog.Accepted:
            self.busy_indicator_window.show();
            #path = dialog.selectedFiles()[0];
            paths = dialog.selectedFiles()[0];
            #Class.data_pool_handler.add_from_dicom_folder(paths);
            self.add_from_dicom_folder_signal.emit(paths);
    
    def select_files_slot(self):
        options = QFileDialog.Options()
        dialog = QFileDialog()
        dialog.setOptions(options)

        dialog.setFilter(dialog.filter() | QtCore.QDir.Hidden)

        dialog.setFileMode(QFileDialog.ExistingFiles)

        dialog.setAcceptMode(QFileDialog.AcceptOpen);
        dialog.setDirectory(QtCore.QDir.currentPath())

        if dialog.exec_() == QDialog.Accepted:
            self.busy_indicator_window.show();
            paths = list(dialog.selectedFiles());
            self.add_from_files_signal.emit(paths);

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

        if self.segmentation_box_tab.currentIndex() == 1:
            if self.foreground_gc_selected is True:
                self.radiograph_view.set_gc_layer_active(True, self.radiograph_view.get_active_layer_index(), 0);
            else:
                self.radiograph_view.set_gc_layer_active(True, self.radiograph_view.get_active_layer_index(), 1);

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
                self.segments_list.setCurrentRow(0);
                idx = self.segments_list.currentRow();
                self.radiograph_view.set_active_layer_index(idx);
                if self.segmentation_box_tab.currentIndex() == 0:
                    self.radiograph_view.set_layer_active(True, idx);
                else:
                    if self.background_gc_selected is True:
                        self.radiograph_view.set_gc_layer_active(True, idx, 1);
                    elif self.foreground_gc_selected is True:
                        self.radiograph_view.set_gc_layer_active(True, idx, 0);
            #If we don't have any layers left, change mouse color to white
            else:
                self.radiograph_view.m_mouse_layer.pen_color = QtGui.QColor(255,255,255);

    def load_finished(self, cnt, show = True):
        self.update_file_info_label();

        #Update the list of available already labeled radiographs
        self.update_all_radiographs_segments_list()

        #if we don't have any images available to show, load one
        if Class.data_pool_handler.current_radiograph == "":
            #Load and set a random image from unlabeled data pool
            pixmap = Class.data_pool_handler.load_unlabeled();
            self.radiograph_label.setText(f"Radiograph Name: {Class.data_pool_handler.current_radiograph}")
            if pixmap is not None:
                self.radiograph_view.set_image(pixmap);
        self.busy_indicator_window.hide();
        if cnt != -1:
            show_dialoge(QMessageBox.Icon.Information, f"Total files added: {cnt}", "Info",QMessageBox.Ok);
        elif cnt == -1:
            if show:
                show_dialoge(QMessageBox.Icon.Information, f"Loaded successfully", "Info",QMessageBox.Ok)

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
        self.fill_button.setStyleSheet("background-color: white")
        self.magnetic_scissor_button.setStyleSheet("background-color: white")
        self.radiograph_view.set_state(LayerItem.DrawState);
    
    def erase_clicked(self):
        self.erase_button.setStyleSheet("background-color: Aquamarine")
        self.paint_button.setStyleSheet("background-color: white")
        self.fill_button.setStyleSheet("background-color: white")
        self.magnetic_scissor_button.setStyleSheet("background-color: white")
        self.radiograph_view.set_state(LayerItem.EraseState);
    
    def erase_gc_clicked(self):
        if self.erase_gc_selected is False:
            self.erase_gc_button.setStyleSheet("background-color: Aquamarine")
            self.radiograph_view.set_state(LayerItem.EraseState);
            self.erase_gc_selected = True;
        else:
            self.erase_gc_button.setStyleSheet("background-color: White")
            self.radiograph_view.set_state(LayerItem.DrawState);
            self.erase_gc_selected = False;

    def fill_clicked(self):
        self.fill_button.setStyleSheet("background-color: Aquamarine");
        self.erase_button.setStyleSheet("background-color: white");
        self.paint_button.setStyleSheet("background-color: white");
        self.magnetic_scissor_button.setStyleSheet("background-color: white");
        self.radiograph_view.set_state(LayerItem.FillState);

    def magnetic_scissor_clicked(self):
        self.fill_button.setStyleSheet("background-color: white");
        self.erase_button.setStyleSheet("background-color: white");
        self.paint_button.setStyleSheet("background-color: white");
        self.magnetic_scissor_button.setStyleSheet("background-color: Aquamarine");
        self.radiograph_view.set_state(LayerItem.MagneticLasso);

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
            names = [];
            for k in df.keys():
                if k != "rot" and k !="exp":
                    names.append(k);
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
            lbl = Class.data_pool_handler.current_radiograph;
            self.predict_on_unlabeled_signal.emit(lbl, Class.data_pool_handler.data_list);
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

            self.radiograph_view.add_layer(layer_names[l], Config.PREDEFINED_COLORS[l]);

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
        Class.data_pool_handler.clear_datalist();
        Class.data_pool_handler.current_radiograph = "";
        self.segments_list.clear();
        self.all_radiographs_list.clear();
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
    
    def save_as_slot(self):
        name = QFileDialog.getSaveFileName(self, 'Save File');
        self.save_project_as_signal.emit(name[0]);
    
    def clahe_value_changed(self):
        radiograph_image = self.radiograph_view.pixels[:,:,:1].squeeze();

        if self.clip_limit_slider.value() > 0 and self.clahe_slider.value() > 0:
            clahe = cv2.createCLAHE(self.clip_limit_slider.value(),(self.clahe_slider.value(),self.clahe_slider.value()));
            radiograph_image = clahe.apply(radiograph_image);
            height, width = radiograph_image.shape

            bytesPerLine = 1 * width
            qImg = QImage(radiograph_image, width, height, bytesPerLine, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qImg);
            self.radiograph_view.set_image(pixmap, reset=False);
        else:
            pixmap = load_radiograph(os.path.sep.join([Config.PROJECT_ROOT, 'images', Class.data_pool_handler.current_radiograph]),
            Class.data_pool_handler.get_current_radiograph_type());
            self.radiograph_view.set_image(pixmap, reset=False);

    def clip_limit_value_changed(self):
        radiograph_image = self.radiograph_view.pixels[:,:,:1].squeeze();
        if self.clip_limit_slider.value() > 0 and self.clahe_slider.value() > 0:
            clahe = cv2.createCLAHE(self.clip_limit_slider.value(),(self.clahe_slider.value(),self.clahe_slider.value()));
            radiograph_image = clahe.apply(radiograph_image);
            height, width = radiograph_image.shape

            bytesPerLine = 1 * width
            qImg = QImage(radiograph_image, width, height, bytesPerLine, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qImg);
            self.radiograph_view.set_image(pixmap, reset=False);
        else:
            pixmap = load_radiograph(os.path.sep.join([Config.PROJECT_ROOT, 'images', Class.data_pool_handler.current_radiograph]),
            Class.data_pool_handler.get_current_radiograph_type());
            self.radiograph_view.set_image(pixmap, reset=False);
    
    def foreground_clicked(self):
        if self.foreground_gc_selected is False:
            self.foreground_button.setStyleSheet("background-color: Aquamarine")
            self.background_button.setStyleSheet("background-color: White")
            self.erase_gc_button.setStyleSheet("background-color: White")
            self.foreground_clicked_signal.emit();
            self.foreground_gc_selected = True;
            self.background_gc_selected = False;
            self.erase_gc_selected = False; 
            self.radiograph_view.set_state(LayerItem.DrawState);
    
    def background_clicked(self):
        if self.background_gc_selected is False:
            self.background_button.setStyleSheet("background-color: Aquamarine")
            self.foreground_button.setStyleSheet("background-color: White")
            self.erase_gc_button.setStyleSheet("background-color: White")
            self.background_clicked_signal.emit();
            self.background_gc_selected = True;
            self.foreground_gc_selected = False;
            self.erase_gc_selected = False;
            self.radiograph_view.set_state(LayerItem.DrawState);
    
    def update_gc_clicked(self):
        self.update_gc_signal.emit();
    
    def reset_gc_clicked(self):
        self.reset_gc_signal.emit();

    def segmentation_tab_changed(self, idx):
        #if we moved to automatic segmentation tab
        if idx == 1:
            self.automatic_segmentation_enable();
            pass
        #if we moved to manual segmentation table
        if idx == 0:
            self.manual_segmentation_enable();
            pass
    
    def update_foreground_with_layer_clicked(self):
        self.update_foreground_with_layer_signal.emit();
    
    def undo_slot(self):
        self.radiograph_view.undo_event();
    
    def redo_slot(self):
        self.radiograph_view.redo_event();

    #--------------------------------------------------------------
    
if __name__=='__main__':


    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    torch.backends.cudnn.benchmark = True
    
    app = QApplication(sys.argv);
    app.setStyleSheet("QPushButton {\
    background-color : white;\
    border-style: outset;\
    border-width: 1px;\
    border-radius: 10px;\
    border-color: beige;\
    font: 10px;\
    min-width: 7em;\
    padding: 10px;\
    }\
    QListView {\
    background-color : HoneyDew;\
    font: 16px;\
    }\
    QPushButton:pressed { background-color: azure; font: 11px;}");

    window = MainWindow();
    new_project_window = NewProjectWindow();

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

    Class.data_pool_handler.load_finished_signal.connect(window.load_finished);
    Class.data_pool_handler.save_project_signal.connect(window.save_project);
    window.start_train_signal.connect(Class.network_trainer.start_train_slot);
    Class.network_trainer.train_finsihed_signal.connect(window.update_model_finished_slot);
    window.save_project_signal.connect(Class.project_handler.save_project);
    Class.project_handler.open_project_signal.connect(Class.data_pool_handler.open_project_slot);
    window.load_model_signal.connect(Class.network_trainer.load_model_for_predicition);
    window.predict_on_unlabeled_signal.connect(Class.network_trainer.predict);
    Class.network_trainer.predict_finished_signal.connect(window.predict_on_unlabeled_finished);
    Class.project_handler.set_project_name_signal.connect(window.set_project_name_slot);
    Class.project_handler.new_project_setup_signal.connect(window.setup_new_project_slot);
    window.open_project_signal.connect(Class.project_handler.open_project);
    new_project_window.open_project_signal.connect(Class.project_handler.open_project);
    window.save_most_recent_project_signal.connect(Class.project_handler.save_most_recent_project);
    window.save_project_as_signal.connect(Class.project_handler.save_project_as);
    window.radiograph_view.size_changed_signal.connect(window.size_value_changed_slot);
    window.update_meta_signal.connect(Class.project_handler.update_project_meta_slot);
    window.get_project_names_signal.connect(Class.project_handler.get_project_names_slot);
    new_project_window.confirm_new_project_clicked_signal.connect(Class.project_handler.new_project);
    window.confirm_new_project_clicked_signal.connect(Class.project_handler.new_project);
    Class.network_trainer.update_train_info_iter.connect(window.trainig_info_window.update_train_info_iter_slot);
    Class.network_trainer.update_valid_info_iter.connect(window.trainig_info_window.update_valid_info_iter_slot);
    Class.network_trainer.update_train_info_epoch_train.connect(window.trainig_info_window.update_train_info_epoch_train_slot);
    Class.network_trainer.update_train_info_epoch_valid.connect(window.trainig_info_window.update_train_info_epoch_valid_slot);
    Class.network_trainer.augmentation_finished_signal.connect(window.trainig_info_window.augmentation_finished_slot);
    Class.network_trainer.model_loaded_finished.connect(window.model_loaded_finished_slot);
    window.foreground_clicked_signal.connect(window.radiograph_view.foreground_clicked_slot);
    window.background_clicked_signal.connect(window.radiograph_view.background_clicked_slot);
    window.update_gc_signal.connect(window.radiograph_view.update_gc_slot);
    window.reset_gc_signal.connect(window.radiograph_view.reset_gc_slot);
    window.update_foreground_with_layer_signal.connect(window.radiograph_view.update_foreground_with_layer_slot);
    window.add_from_files_signal.connect(Class.data_pool_handler.add_from_files_slot);
    window.add_from_folder_signal.connect(Class.data_pool_handler.add_from_folder_slot);
    window.add_from_dicom_folder_signal.connect(Class.data_pool_handler.add_from_dicom_folder_slot);

    ret = Class.project_handler.open_project();
    
    if not ret:
        #No project found so show project name dialoge.
       
        new_project_window.show_window();
        pass

    sys.exit(app.exec_());