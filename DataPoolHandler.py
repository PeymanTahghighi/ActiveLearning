#==================================================================
#==================================================================
import os
from posixpath import basename
from typing import Dict
from PyQt5 import QtWidgets
from PyQt5 import QtCore
import cv2
from PyQt5 import QtGui
from ignite import base
from pandas.core.frame import DataFrame
import pickle
from Utility import *
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal
import numpy as np
from shutil import copyfile
from glob import glob
import pandas as pd
import Config
#==================================================================
#==================================================================

#------------------------------------------------------------------
class DataPoolHandler(QObject):
    load_finished_signal = pyqtSignal();
    save_project_signal = pyqtSignal(bool);

    def __init__(self):
        super().__init__();
        self.num_data = 0;
        self.data_list = dict();
        self.main_window = None;
        self.__current_radiograph = "";
        
        pass

    @property 
    def current_radiograph(self):
        return self.__current_radiograph;

    @current_radiograph.setter
    def current_radiograph(self, c):
        self.__current_radiograph = c;
    
    def clear_datalist(self):
        self.data_list.clear();
    
    def set_main_window(self, mw):
        self.main_window = mw;
    
    def add_from_files(self,paths):
        cnt = 0;
        for p in paths:
            img_name = os.path.basename(p);
            #Check for duplicate file names
            if img_name in self.data_list.keys():
                ext = img_name[img_name.rfind('.'):];
                img_name_wo_ext = img_name[:img_name.rfind('.')];
                while img_name in self.data_list.keys():
                    img_name_wo_ext += '_1';
                    img_name = img_name_wo_ext + ext;
                
            self.data_list[img_name] = ["unlabeled"];

            #Copy all files to project path
            file_new_path = os.path.sep.join([Config.PROJECT_ROOT, 'images', img_name]);
            copyfile(p,file_new_path);
            cnt += 1;
        show_dialoge(QMessageBox.Icon.Information, f"Total files added: {cnt}", "Info",QMessageBox.Ok)
        self.load_finished_signal.emit();
        self.save_project_signal.emit(False);
        pass
    
    def add_from_folder(self, folder_path):
        lst = glob(folder_path + "/*");
        cnt=0;
        for p in lst:
            img_name = os.path.basename(p);
            #Check for duplicate file names
            if img_name in self.data_list.keys():
                ext = img_name[img_name.rfind('.'):];
                img_name_wo_ext = img_name[:img_name.rfind('.')];
                while img_name in self.data_list.keys():
                    img_name_wo_ext += '_1';
                    img_name = img_name_wo_ext + ext;
                
            self.data_list[img_name] = ["unlabeled"];
            
            #Copy all files to project path
            file_new_path = os.path.sep.join([Config.PROJECT_ROOT, 'images', img_name]);
            copyfile(p,file_new_path);
            cnt += 1;
        show_dialoge(QMessageBox.Icon.Information, f"Total files added: {cnt}", "Info",QMessageBox.Ok)
        self.load_finished_signal.emit();
        self.save_project_signal.emit(False);
    
    def get_all_unlabeled(self):
        unlabeled = [];
        for key in self.data_list.keys():
            if self.data_list[key][0] == 'unlabeled':
                unlabeled.append(key);
        return unlabeled;
    
    '''
        This function loads a radiograph from disk with all layers
        name should be the exact name on the disk with extensions so for meta
        we extract file name first
    '''
    def load_radiograph(self, path):
        mask_meta_path = os.path.basename(path);
        mask_meta_path = mask_meta_path[:mask_meta_path.rfind('.')];
        mask_meta_path = os.path.sep.join([Config.PROJECT_ROOT, 'labels', mask_meta_path+".meta"])
        path = os.path.sep.join([Config.PROJECT_ROOT, 'images', path]);
        r,m = load_radiograph_masks(path,mask_meta_path);
        return r, m;

    def get_all_labeled(self):
        labeled = [];
        for key in self.data_list.keys():
            if self.data_list[key][0] == 'labeled':
                labeled.append(key);
        return labeled;

    def load_random_unlabeled(self):
        unlabeled = self.get_all_unlabeled();  
        if len(unlabeled) == 0:
            show_dialoge(QMessageBox.Icon.Information,
             f"No unlabeled radiographs found. Please add new radiographs or selected images already labaled from the list", 
             "No radiographs found",QMessageBox.Ok);
            return None;
        #Randomly select one data
        if len(self.data_list) != 1:
            r = np.random.randint(0,len(unlabeled));
            p = os.path.sep.join([Config.PROJECT_ROOT, 'images', unlabeled[r]]);
            pixmap = QtGui.QPixmap(p);
            if pixmap.isNull():
                    print('Cannot open')
                    return
            self.__current_radiograph = unlabeled[r];
        else:
            p = os.path.sep.join([Config.PROJECT_ROOT, 'images', unlabeled[0]]);
            pixmap = QtGui.QPixmap(p);
            self.__current_radiograph = unlabeled[0];
        
        return pixmap;

    def next_unlabeled(self):
        unlabeled = self.get_all_unlabeled();

        if len(unlabeled) == 0:
            show_dialoge(QMessageBox.Icon.Information,
             f"No unlabeled radiographs found. Please add new radiographs or selected images already labaled from the list", 
             "No radiographs found",QMessageBox.Ok);
            return None;
        
        #Randomly select one data
        r = np.random.randint(0,len(unlabeled));
        while unlabeled[r] == self.__current_radiograph:
            r = np.random.randint(0,len(unlabeled));
        p = os.path.sep.join([Config.PROJECT_ROOT, 'images', unlabeled[r]]);
        pixmap = QtGui.QPixmap(p);
        self.__current_radiograph = unlabeled[r];

        return pixmap;

    def submit_label(self,arr):
        self.data_list[self.__current_radiograph][0] = "labeled";

        path_tmp = self.__current_radiograph.replace('\\','/');
        #save label to labels folder and save meta data about radiograph
        file_name = os.path.basename(path_tmp);
        file_name = file_name[0:file_name.find('.')];
        data_dict = dict();

        for l in range(len(arr)):
            layer = arr[l][0];
            layer_name = arr[l][1];

            rdg = layer.get_numpy();
            clr = layer.pen_color.name();
            path_to_file = os.path.sep.join([Config.PROJECT_ROOT, "labels", file_name + f"_{l}.png"]);
            cv2.imwrite(path_to_file, rdg);
            data_dict[layer_name] = [f'{l}', clr,  file_name + f"_{l}.png"];
        
        p = os.path.sep.join([Config.PROJECT_ROOT,"labels",file_name + ".meta"]);

        pickle.dump(data_dict,open(p, 'wb'));

        show_dialoge(QMessageBox.Icon.Information, f"Label successfully submitted.", "Info",QMessageBox.Ok);
        pass
    
    def open_project_slot(self, dc, show = True):
        self.data_list = dc;
        if show:
            show_dialoge(QMessageBox.Icon.Information, f"Loaded successfully", "Info",QMessageBox.Ok)
        self.load_finished_signal.emit();
#------------------------------------------------------------------