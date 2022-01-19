#==================================================================
#==================================================================
from genericpath import isdir, isfile
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
import pydicom
from pydicom.fileset import FileSet
#==================================================================
#==================================================================

#------------------------------------------------------------------
class DataPoolHandler(QObject):
    load_finished_signal = pyqtSignal();
    save_project_signal = pyqtSignal(bool);

    def __init__(self):
        super().__init__();
        self.__data_list = dict();
        self.__current_radiograph = "";

    @property 
    def current_radiograph(self):
        return self.__current_radiograph;

    @current_radiograph.setter
    def current_radiograph(self, c):
        self.__current_radiograph = c;

    @property
    def data_list(self):
        return self.__data_list;
    
    def get_current_radiograph_type(self):
        return self.__data_list[self.__current_radiograph][1];
    
    def clear_datalist(self):
        self.__data_list.clear();
    
    def add_from_files(self, paths):
        """
            This function loads the files indicated by the 'paths' parameter.
        """
        cnt = self.__load_file_paths(paths);
        show_dialoge(QMessageBox.Icon.Information, f"Total files added: {cnt}", "Info",QMessageBox.Ok)
        self.load_finished_signal.emit();
        self.save_project_signal.emit(False);
        pass
    
    
    def add_from_folder(self, folder_path):
        """
            Adds data from folder. This function reads an entire folder and find image format 
            files such as "jpg", "png" and so on. It only loads common image formats. We use 
            seperate function to load dicom folder.
        """
        cnt = self.__load_folder(folder_path);
        show_dialoge(QMessageBox.Icon.Information, f"Total files added: {cnt}", "Info",QMessageBox.Ok)
        self.load_finished_signal.emit();
        self.save_project_signal.emit(False);
    
    def add_from_dicom_folder(self, folder_path):
        """
            Adds data from folder. This function reads an entire folder and search
            fold dicomdir file. This file contain all information about dicom files
            in this directory. If this file doesn't exsist, it means that this directory
            is not a dicom directory.
        """
        cnt = self.__load_dicom_folder(folder_path);
        show_dialoge(QMessageBox.Icon.Information, f"Total files added: {cnt}", "Info",QMessageBox.Ok);
        self.load_finished_signal.emit();
        self.save_project_signal.emit(False);
    
    def get_all_unlabeled(self):
        unlabeled = [];
        for key in self.__data_list.keys():
            if self.__data_list[key][0] == 'unlabeled':
                unlabeled.append([key, self.__data_list[key][1]]);
        return unlabeled;
    
    '''
        This function loads a radiograph from disk with all layers
        name should be the exact name on the disk with extensions so for meta
        we extract file name first
    '''
    def load_radiograph(self, name):
        mask_meta_path = name[:name.rfind('.')];
        mask_meta_path = os.path.sep.join([Config.PROJECT_ROOT, 'labels', mask_meta_path+".meta"])
        path = os.path.sep.join([Config.PROJECT_ROOT, 'images', name]);
        radiograph_type = self.__data_list[name][1];
        if os.path.exists(mask_meta_path):
            r,m = load_radiograph_masks(path,mask_meta_path, radiograph_type);
            return r, m;
        else:
            r = load_radiograph(path, radiograph_type);
            return r, list();

    def get_all_labeled(self):
        labeled = [];
        for key in self.__data_list.keys():
            if self.__data_list[key][0] == 'labeled':
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
        if len(self.__data_list) != 1:
            r = np.random.randint(0,len(unlabeled));
            p = os.path.sep.join([Config.PROJECT_ROOT, 'images', unlabeled[r][0]]);
            pixmap = load_radiograph(p, unlabeled[r][1]);
            if pixmap.isNull():
                    print('Cannot open')
                    return
            self.__current_radiograph = unlabeled[r][0];
        else:
            p = os.path.sep.join([Config.PROJECT_ROOT, 'images', unlabeled[0][0]]);
            pixmap = load_radiograph(p, unlabeled[0][1]);
            self.__current_radiograph = unlabeled[0][0];
        
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
        while unlabeled[r][0] == self.__current_radiograph:
            r = np.random.randint(0,len(unlabeled));
        
        p = os.path.sep.join([Config.PROJECT_ROOT, 'images', unlabeled[r][0]]);
        pixmap = load_radiograph(p, unlabeled[r][1]);
        self.__current_radiograph = unlabeled[r][0];

        return pixmap;
    
    def delete_radiograph(self, txt):
        self.__data_list.pop(txt);

    def submit_label(self,  arr, rot, exp):
        self.__data_list[self.__current_radiograph][0] = "labeled";

        path_tmp = self.__current_radiograph.replace('\\','/');
        #save label to labels folder and save meta data about radiograph
        file_name = os.path.basename(path_tmp);
        file_name = file_name[0:file_name.find('.')];
        data_dict = dict({'rot' : rot, 'exp': exp});

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
        self.__data_list = dc;
        if show:
            show_dialoge(QMessageBox.Icon.Information, f"Loaded successfully", "Info",QMessageBox.Ok)
        self.load_finished_signal.emit();
    

    #*****************
    #Private functions
    #*****************

    def __is_dicom(self, path):
        """
            This function checks if a given path is dicom or not using
            file extension and checking file data
        """
        _, ext = os.path.splitext(path);
        #Here, we assume that if the dataset size is one,
        #the data is NOT in dicom format, otherwise it is
        ds = pydicom.dcmread(path, force=True);
        
        if ext == ".dcm" or len(ds) != 1:
            return True;
        return False;
    
    def __is_image(self, path):
        """
            This function checks if a given file can be loaded
            by this program or not.
        """
        img = cv2.imread(path);

        #we check image size as a sign of successfull or failed loading of the image file.
        if img.size == 0:
            return False;
        return True;
    
    def __add_image_to_database(self, item_path, item_name):
        """
            This function first check if we can load this file
            and it is in common formats for this program. Then add
            this image to the database of all images that this project
            possess.
        """
        if(self.__is_dicom(item_path)):
            item_name = self.__check_duplicate_name(item_name);
            #save as a dicom image
            self.__data_list[item_name] = ["unlabeled", "dicom"];
            file_new_path = os.path.sep.join([Config.PROJECT_ROOT, 'images', item_name]);
            copyfile(item_path, file_new_path);
            return 1;
        elif(self.__is_image(item_path)):
            item_name = self.__check_duplicate_name(item_name);
            #save as a dicom image
            self.__data_list[item_name] = ["unlabeled", "image"];
            file_new_path = os.path.sep.join([Config.PROJECT_ROOT, 'images', item_name]);
            copyfile(item_path, file_new_path);
            return 1;
        
        return 0;

    def __check_duplicate_name(self, name):
        """
            This function checks for duplicate file names and add 
            indices like "_#_ which # is a number to the end of
            the given file name
        """
        if name in self.__data_list.keys():
            img_name_wo_ext, ext = os.path.splitext(name);
            idx = 1;
            #loop until we find a unique name for this file
            while name in self.__data_list.keys():
                img_name_wo_ext += f'_{idx}';
                name = img_name_wo_ext + ext;
                idx += 1;
        
        return name;

    def __load_dicom_folder(self, folder_path):
        """
            In this function we use os.walk function to get all files
            in the directory and then we search for "DICOMDIR" file.
            If this file doens't exists it means that this directory
            is not a dicom folder. Otherwise, we load dicomdir which 
            contains all dicom files in this directory.
            For reading dicomdir, we load all instance and then we use
            save_as function to save it to the location that we want
            inside project directory.
        """
        cnt = 0;
        lst = os.walk(folder_path);
        has_dicomdir = False;
        dicomdir_file_path = "";
        dicom_file_name = "";
        #search for the dicom file
        for root, dir, entry in lst:
            for item_name in entry:
                if item_name == "DICOMDIR":
                    has_dicomdir = True;
                    dicomdir_file_path = f"{root}\\{item_name}";
                    #we use the name of the directory as the name of the file
                    dicom_file_name_root = root[root.replace("\\","/").rfind("/")+1:];
                    break;

            if has_dicomdir is True:
                break;

        if(has_dicomdir):
            ds = FileSet(dicomdir_file_path);
            for instance in ds:
                dicom = instance.load();
                dicom_file_name = self.__check_duplicate_name(dicom_file_name_root);
                #add actual data to the list
                self.__data_list[dicom_file_name] = ["unlabeled", "dicom"];
                #save as inside project directory.
                file_new_path = os.path.sep.join([Config.PROJECT_ROOT, 'images', dicom_file_name]);
                dicom.save_as(file_new_path);
                cnt += 1;
        return cnt;
    

    def __load_folder(self, folder_path):
        """
            This function search for all image format in the given directory and
            load them.
            This function does NOT load dicom files. 
        """
        cnt = 0;
        lst = os.walk(folder_path);
        for root, dir, entry in lst:
            for item_name in entry:
                item_path = f"{root}\\{item_name}";
                cnt += self.__add_image_to_database(item_path, item_name);
        return cnt;
    
    def __load_file_paths(self, paths):
        cnt = 0;
        for p in paths:
            item_name = self.__check_duplicate_name(os.path.basename(p));
            cnt += self.__add_image_to_database(p, item_name);
        
        return cnt;
#------------------------------------------------------------------