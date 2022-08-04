
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox
from PyQt5.QtGui import QIcon, QImage, QPixmap
import os
import pickle

import cv2
import Config
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
import numpy as np

def show_dialoge(icon, text, title, buttons, parent = None):
    if parent != None:
        msgBox = QMessageBox(parent);
    else:
        msgBox = QMessageBox()
    msgBox.setIcon(icon)
    msgBox.setText(text)
    msgBox.setWindowTitle(title)
    msgBox.setStandardButtons(buttons)

    returnValue = msgBox.exec()

def get_radiograph_label_meta(radiograph_root, mask_root):
    radiograph_list = os.listdir(radiograph_root);
    mask_list = os.listdir(mask_root);

    mask_names = [];
    radiograph_names = [];

    for m in mask_list:
        if m.find('meta') != -1:
            #Find file extension
            ext = "";
            file_name,_ = os.path.splitext(m);
            for r in radiograph_list:
                n,ext= os.path.splitext(r);
                if n == file_name:
                    break;
            mask_names.append(os.path.sep.join([mask_root,m]));
            radiograph_names.append(os.path.sep.join([radiograph_root,file_name+ext]));
    
    return radiograph_names, mask_names;

def load_radiograph_masks(radiograph_path, mask_path, type):
    radiograph_pixmap = load_radiograph(radiograph_path, type);
    #Open each mask in meta data and add them to list
    df = pickle.load(open(mask_path,'rb'));
    mask_pixmap_list = [];
    for k in df.keys():
		if k != 'rot' and k != 'exp' and k!= 'misc':
            p = df[k][2];
            path = os.path.sep.join([Config.PROJECT_ROOT, 'labels', p]);
            mask_pixmap = QtGui.QPixmap(path);
            mask_pixmap_list.append([k,df[k][1],mask_pixmap]);

    return radiograph_pixmap, mask_pixmap_list;

def load_radiograph(radiograph_path, type, return_type = 'pixmap', imread_type = cv2.IMREAD_GRAYSCALE):
    if type=='dicom':
        ds = pydicom.dcmread(radiograph_path, force=True);
        pix_arr = ds.pixel_array;
        

        pix_arr = apply_modality_lut(pix_arr, ds);
        w = apply_voi_lut(pix_arr, ds);

        if ds['PhotometricInterpretation'].repval == "'MONOCHROME1'":
            w = np.amax(w) - w;

        w = w - w.min();
        image_2d_scaled = (np.maximum(w,0) / w.max()) * 255.0
        image_2d_scaled = np.uint8(image_2d_scaled);
        
        if return_type != 'pixmap':
            return image_2d_scaled;
        height, width = image_2d_scaled.shape;
        bytesPerLine = 1 * width
        qImg = QImage(image_2d_scaled, width, height, bytesPerLine, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qImg);
    else:
        if return_type == 'pixmap':
            pixmap = QPixmap(radiograph_path);
        else:
            pixmap = cv2.imread(radiograph_path, imread_type);

    return pixmap;
