#==================================================================
#==================================================================

import os
import cv2
import pickle
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QObject,  pyqtSignal
import numpy as np
from shutil import copyfile
#import ptvsd
#import ptvsd
#import ptvsd
##import ptvsd
import pydicom
from pydicom import dcmread
from Utility import *
import Config
from Strategy import get_grad_embeddings, get_cluster_centers
from utils import JSD
#import ptvsd
from pathlib import Path
#==================================================================
#==================================================================

#------------------------------------------------------------------
class DataPoolHandler(QObject):
    load_finished_signal = pyqtSignal(int, bool);
    save_project_signal = pyqtSignal(bool);

    def __init__(self):
        super().__init__();
        self.__data_list = dict();
        #this is only for compatibility issues, we can remove this later on
        self.__data_list_hist = dict();
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
    
    def add_from_files_slot(self, paths):
        """
            This function loads the files indicated by the 'paths' parameter.
        """
        #ptvsd.debug_this_thread();
        cnt = self.__load_file_paths(paths);
        self.save_project_signal.emit(False);
        self.load_finished_signal.emit(cnt, True);
        pass
    
    
    def add_from_folder_slot(self, folder_path):
        """
            Adds data from folder. This function reads an entire folder and find image format 
            files such as "jpg", "png" and so on. It only loads common image formats. We use 
            seperate function to load dicom folder.
        """
        cnt = self.__load_folder(folder_path);
        self.save_project_signal.emit(False);
        self.load_finished_signal.emit(cnt, True);
    
    def add_from_dicom_folder_slot(self, folder_path):
        """
            Adds data from folder. This function reads an entire folder and search
            fold dicomdir file. This file contain all information about dicom files
            in this directory. If this file doesn't exsist, it means that this directory
            is not a dicom directory.
        """
        cnt = self.__load_dicom_folder(folder_path);
        self.save_project_signal.emit(False);
        self.load_finished_signal.emit(cnt, True);
    
    def get_all(self, type = 'unlabeled'):
        ret = [];
        for key in self.__data_list.keys():
            if self.__data_list[key][0] == type:
                ret.append([key, self.__data_list[key][1], self.__data_list[key][2]]);
        return ret;
    
    '''
        This function loads a radiograph from disk with all layers
        name should be the exact name on the disk with extensions so for meta
        we extract file name first
    '''
    def load_radiograph(self, name):
       # ptvsd.debug_this_thread();
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

    def load_unlabeled(self):

        unlabeled = self.get_all(); 

        if len(unlabeled) == 0:
                self.current_radiograph == '';
                show_dialoge(QMessageBox.Icon.Information,
                f"No unlabeled radiographs found. Please add new radiographs or selected images already labaled from the list", 
                "No radiographs found",QMessageBox.Ok);
                return None;
        
        if Config.NEXT_SAMPLE_SELECTION == 'Random':
            r = np.random.randint(0,len(unlabeled));
            p = os.path.sep.join([Config.PROJECT_ROOT, 'images', unlabeled[r][0]]);
            pixmap = load_radiograph(p, unlabeled[r][1]);
            self.__current_radiograph = unlabeled[r][0];
        elif Config.NEXT_SAMPLE_SELECTION == 'Similarity':
            pixmap = self.__get_next_similarity();
        
        return pixmap;

    def next_unlabeled(self):
        '''
            Here we apply our data selection strategy.
        '''
        unlabeled = self.get_all(type = 'unlabeled');

        if len(unlabeled) == 0:
            self.current_radiograph == '';
            show_dialoge(QMessageBox.Icon.Information,
             f"No unlabeled radiographs found. Please add new radiographs or selected images already labaled from the list", 
             "No radiographs found",QMessageBox.Ok);
            return None;

        #get model first. If no moodel exists, then we will use random sampling.
        #This is indeed useful in first step when we haven't trained any models 
        #yet or in situation that we lost the model.
        #TODO implement badge gradient sampling
        #m, sts = Class.network_trainer.get_model();

        # if sts is True:
        #     idx = self.__get_data_badge_strategy(unlabeled, m);
        # else:
        #random data sampling
        if Config.NEXT_SAMPLE_SELECTION == 'Random':
            idx = np.random.randint(0,len(unlabeled));
            while unlabeled[idx][0] == self.__current_radiograph:
                idx = np.random.randint(0,len(unlabeled));
            
            p = os.path.sep.join([Config.PROJECT_ROOT, 'images', unlabeled[idx][0]]);
            pixmap = load_radiograph(p, unlabeled[idx][1]);
            self.__current_radiograph = unlabeled[idx][0];
        elif Config.NEXT_SAMPLE_SELECTION == 'Similarity':
            pixmap = self.__get_next_similarity();
        return pixmap;
    
    def delete_radiograph(self, txt):
        self.__data_list.pop(txt);
        if os.path.exists(os.path.sep.join([Config.PROJECT_ROOT, 'images', txt])):
            os.remove(os.path.sep.join([Config.PROJECT_ROOT, 'images', txt]));

        #delete labels if exists
        file_name = txt[:txt.rfind('.')];
        if os.path.exists(os.path.join(Config.PROJECT_ROOT, 'labels', f'{file_name}.meta')):
            meta_file = pickle.load(open(os.path.join(Config.PROJECT_ROOT, 'labels', f'{file_name}.meta'), 'rb'));
            for k in meta_file.keys():
                if k != 'misc' and k!='rot' and k!='exp':
                    os.remove(os.path.join(Config.PROJECT_ROOT, 'labels',meta_file[k][2]));
        
            os.remove(os.path.join(Config.PROJECT_ROOT, 'labels', f'{file_name}.meta'));
        
        self.save_project_signal.emit(False);

    def submit_label(self,  arr, misc):
        #ptvsd.debug_this_thread();
        self.__data_list[self.__current_radiograph][0] = "labeled";

        path_tmp = self.__current_radiograph.replace('\\','/');
        #save label to labels folder and save meta data about radiograph
        file_name = os.path.basename(path_tmp);
        file_name = file_name[0:file_name.rfind('.')];
        data_dict = dict({'misc' : misc});

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
        self.__update_histograms(dc);
        self.load_finished_signal.emit(-1, show);
    
    def rename_file(self, orig_name, new_name):
        
        #ptvsd.debug_this_thread();
        
        _,ext = os.path.splitext(orig_name);
        if f"{new_name}{ext}" in self.__data_list.keys():
            return False;

        self.__data_list[f"{new_name}{ext}"] = self.__data_list[orig_name];
        self.__data_list.pop(orig_name);
        orig_name_we = orig_name[:orig_name.rfind('.')];

        #rename image, if image with new name already exists, skip renaming
        if os.path.exists(f'{Config.PROJECT_ROOT}\\images\\{new_name}{ext}') is False:
            os.rename(f'{Config.PROJECT_ROOT}\\images\\{orig_name}', f'{Config.PROJECT_ROOT}\\images\\{new_name}{ext}');
        #remove original image
        if os.path.exists(f'{Config.PROJECT_ROOT}\\images\\{orig_name}') is True:
            os.remove(f'{Config.PROJECT_ROOT}\\images\\{orig_name}');

        if self.__data_list[f"{new_name}{ext}"][0] == 'labeled':
            #rename all_labels
            meta_file = pickle.load(open(os.path.join(Config.PROJECT_ROOT, 'labels', f'{orig_name_we}.meta'), 'rb'));
            for m in meta_file.keys():
                if m != 'misc' and m!= 'rot' and m!='exp':
                    mask_name = meta_file[m][2];
                    mask_idx = mask_name[mask_name.find('_'):];
                    new_mask_name = f"{new_name}{mask_idx}";
                    meta_file[m][2] = new_mask_name;

                    #rename mask
                    if os.path.exists(os.path.join(Config.PROJECT_ROOT, 'labels', new_mask_name)) is False:
                        os.rename(os.path.join(Config.PROJECT_ROOT, 'labels', mask_name), os.path.join(Config.PROJECT_ROOT, 'labels', new_mask_name));
            
            #rename meta file
            os.remove(f'{Config.PROJECT_ROOT}\\labels\\{orig_name_we}.meta');
            pickle.dump(meta_file, open(f'{Config.PROJECT_ROOT}\\labels\\{new_name}.meta', 'wb'));
        
        self.save_project_signal.emit(False);

        return True;
    

    def rename_layer(self, orig_name, new_name):
        
        #ptvsd.debug_this_thread();
        file_name = self.__current_radiograph[:self.__current_radiograph.rfind('.')];
        meta_file = pickle.load(open(os.path.join(Config.PROJECT_ROOT, 'labels', f'{file_name}.meta'), 'rb'));
        if new_name in meta_file.keys():
            return False;
        meta_file[new_name] = meta_file[orig_name];
        meta_file.pop(orig_name);
        pickle.dump(meta_file, open(os.path.join(Config.PROJECT_ROOT,'labels', f'{file_name}.meta'), 'wb'));
        
        self.save_project_signal.emit(False);

        return True;
    

    #*****************
    #Private functions
    #*****************

    def __update_histograms(self, dl):
        '''
            Here we update histograms if we have not calculated for any images
            This way, we can have backward compatibility.
        '''
        #ptvsd.debug_this_thread();
        change = False;
        for k in dl.keys():
            #if we only have two items for each image,
            #it basically means that we don't have any histograms
            if len(dl[k]) == 2:
                pixmap = load_radiograph(os.path.join(Config.PROJECT_ROOT, 'images', k), dl[k][1], 'array');
                hist = cv2.calcHist([pixmap], [0], None, [256], [0,255]);
                dl[k].append(hist/hist.sum());
                change = True;
        
        self.__data_list =  dl;
        if change is True:
            self.save_project_signal.emit(False);
        
        return dl;

    def __get_next_similarity(self):
        labeled = self.get_all('labeled'); 
        unlabeled = self.get_all();

        #if we don't have any labeled images, return one ranomly
        if len(labeled) == 0:
            idx = np.random.randint(0,len(unlabeled));
            p = os.path.sep.join([Config.PROJECT_ROOT, 'images', unlabeled[idx][0]]);
            pixmap = load_radiograph(p, unlabeled[idx][1]);
            self.__current_radiograph = unlabeled[idx][0];

            return pixmap;

        max_d = 0;
        selected_idx = 0;
        for idx, uh in enumerate(unlabeled):
            min_d = 100;
            for lh in labeled:
                dist = JSD(uh[2], lh[2]);
                if dist < min_d:
                    min_d = dist;
            
            if min_d > max_d:
                max_d = min_d;
                selected_idx = idx;


        p = os.path.sep.join([Config.PROJECT_ROOT, 'images', unlabeled[selected_idx][0]]);
        pixmap = load_radiograph(p, unlabeled[selected_idx][1]);
        self.__current_radiograph = unlabeled[selected_idx][0];

        return pixmap;
        
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
            if item_name not in self.__data_list.keys():
                #save as a dicom image
                pixmap = load_radiograph(item_path, 'dicom', return_type='array');
                hist = cv2.calcHist([pixmap], [0], None, [256], [0,255]);
                self.__data_list[item_name] = ["unlabeled", "dicom", hist/hist.sum()];
                file_new_path = os.path.sep.join([Config.PROJECT_ROOT, 'images', item_name]);
                copyfile(item_path, file_new_path);
                return 1;
        elif(self.__is_image(item_path)):
            if item_name not in self.__data_list.keys():
                pixmap = load_radiograph(item_path, 'image', return_type='array');
                hist = cv2.calcHist([pixmap], [0], None, [256], [0,255]);
                self.__data_list[item_name] = ["unlabeled", "image", hist/hist.sum()];
                file_new_path = os.path.sep.join([Config.PROJECT_ROOT, 'images', item_name]);
                copyfile(item_path, file_new_path);
                return 1;
        
        return 0;

    def __check_duplicate_name(self, name):
        """
            This function checks for duplicate file names and add 
            indices like _#_ which # is a number to the end of
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
        #ptvsd.debug_this_thread();
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
            dicomdir_file_root = dicomdir_file_path[:dicomdir_file_path.rfind('\\')];
            ds = dcmread(dicomdir_file_path)
            # Iterate through the PATIENT records
            for patient in ds.patient_records:

                # Find all the STUDY records for the patient
                studies = [
                    ii for ii in patient.children if ii.DirectoryRecordType == "STUDY"
                ]
                for study in studies:
                    # Find all the SERIES records in the study
                    all_series = [
                        ii for ii in study.children if ii.DirectoryRecordType == "SERIES"
                    ]
                    for series in all_series:
                        # Find all the IMAGE records in the series
                        images = [
                            ii for ii in series.children
                            if ii.DirectoryRecordType == "IMAGE"
                        ]
                        
                        # Get the absolute file path to each instance
                        #   Each IMAGE contains a relative file path to the root directory
                        elems = [ii["ReferencedFileID"] for ii in images]
                        # Make sure the relative file path is always a list of str
                        paths = [[ee.value] if ee.VM == 1 else ee.value for ee in elems]
                        paths = [Path(*p) for p in paths]

                        # List the instance file paths
                        for p in paths:
                            p = os.fspath(p);
                            file_name = p[p.rfind('\\')+1:];
                            data_list_name = f"{dicom_file_name_root}_{file_name}";
                            if data_list_name not in self.__data_list:
                                #save as a dicom image
                                pixmap = load_radiograph(os.path.join(dicomdir_file_root, p), 'dicom', return_type='array');
                                hist = cv2.calcHist([pixmap], [0], None, [256], [0,255]);
                                self.__data_list[data_list_name] = ["unlabeled", "dicom", hist/hist.sum()];
                                file_new_path = os.path.sep.join([Config.PROJECT_ROOT, 'images', data_list_name]);
                                copyfile(os.path.join(dicomdir_file_root, p), file_new_path);
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
            #ignore duplicate names
            if os.path.basename(p) not in self.__data_list.keys():
                item_name = os.path.basename(p);
                #item_name = self.__check_duplicate_name(os.path.basename(p));
                cnt += self.__add_image_to_database(p, item_name);
        return cnt;
    def __get_data_badge_strategy(self, unlabeled, model):
        X = [];
        for entry in unlabeled:
            X.append(os.path.sep.join([Config.PROJECT_ROOT, "images", entry[0]]));
        grad_embeddings = get_grad_embeddings(X, model);
        idx = get_cluster_centers(grad_embeddings, 1, None);
        return idx;


#------------------------------------------------------------------