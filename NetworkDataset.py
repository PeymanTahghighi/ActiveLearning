from PyQt5 import QtCore
from pandas.io import pickle
from Utility import get_radiograph_label_meta
from PIL import ImageColor, Image
import PIL
from imgaug.augmenters.meta import Sometimes
import numpy as np
import os
from numpy.lib.function_base import copy
from numpy.lib.type_check import imag
from scipy.sparse.construct import random
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import cv2
from sklearn.utils import shuffle
import torch
from glob import glob
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import Config
import logging

from utils import JSD

class TrainValidSplit():
    def __init__(self):
        pass

    def get(self, root_radiograph, root_mask, valid_split, layers_names):
        self.radiograph_root = root_radiograph;
        self.mask_root = root_mask;
        self.mask_names = [];
        self.radiograph_names = [];

        all_radiograph_names, all_mask_names = get_radiograph_label_meta(root_radiograph, root_mask);
        selected_radiographs = [];
        selected_masks = [];

        #Here, we first find images that has all the user selected layers
        for m in range(len(all_mask_names)):
            df = pickle.load(open(all_mask_names[m],'rb'));
            df_keys = df.keys();
            
            add = True;
            for l in layers_names:
                if l not in df_keys:
                    add = False;
                    break;

            #If this image has all the layers wanted, add it to the list of radiographs
            if add is True:
                selected_masks.append(all_mask_names[m]);
                selected_radiographs.append(all_radiograph_names[m]);
        
        #calculate histogram of all selected radiographs once
        selected_radiographs_hist = [];
        for sr in selected_radiographs:
            selected_radiographs_hist.append(cv2.calcHist([cv2.imread(sr, cv2.IMREAD_GRAYSCALE)],[0], None, [256], [0,256] ))

        selected_indices = [];
        selected_index_fir_step = np.random.randint(0,len(selected_radiographs_hist));
        selected_indices.append(selected_index_fir_step);
        hist_r = selected_radiographs_hist[selected_index_fir_step];
        hist_r = hist_r / hist_r.sum();
        max_dist = 0;
        selected_index_sec_step = 0;
        for i in range(len(selected_radiographs_hist)):
            if i != selected_index_fir_step:
                hist_i = selected_radiographs_hist[i];
                hist_i = hist_i / hist_i.sum();
                dist = JSD(hist_i, hist_r);
                if dist > max_dist:
                    max_dist = dist;
                    selected_index_sec_step = i;
        
        selected_indices.append(selected_index_sec_step);

        dataset_size = len(selected_radiographs_hist);
        train_data = int(np.ceil((1-valid_split) * dataset_size)) - 2;

        while(train_data != 0):
            max_dist = 0;
            selected_idx = 0;
            for i in range(len(selected_radiographs_hist)):
                hist_i = selected_radiographs_hist[i];
                hist_i = hist_i / hist_i.sum();
                min_dist = float('inf');
                for sel in selected_indices:
                    if i != sel:
                        hist_sel = selected_radiographs_hist[sel];
                        hist_sel = hist_sel / hist_sel.sum();
                        dist = JSD(hist_sel, hist_i);
                        if dist < min_dist:
                            min_dist = dist;
                    
                if min_dist > max_dist and i not in selected_indices:
                    max_dist = min_dist;
                    selected_idx = i;
            
            selected_indices.append(selected_idx);
            train_data -= 1;
        
                    

        #Split datat into train and validation
        selected_radiographs = np.array(selected_radiographs);
        selected_masks = np.array(selected_masks);

        print(f"[TRAIN INFO] | Train size(before augmentation): {len(selected_indices)} \tValid size: {dataset_size - len(selected_indices)}");

        train_radiograph_filenames = selected_radiographs[selected_indices];
        train_mask_filenames = selected_masks[selected_indices];

        valid_indices = list(set(np.arange(0,dataset_size)) -  set(selected_indices));

        valid_radiograph_filenames = selected_radiographs[valid_indices]
        valid_mask_filenames = selected_masks[valid_indices]

        return train_radiograph_filenames, train_mask_filenames, valid_radiograph_filenames, valid_mask_filenames;

class OfflineAugmentation():
    def __init__(self):
        self.rotation_range = np.arange(-20,20,1,dtype=np.float);
        self.brightness_range = np.arange(0.5,1.8,0.1,dtype=np.float);
        sometimes = lambda aug: iaa.Sometimes(0.5, aug);
        self.seq = iaa.Sequential([
            #iaa.Affine(rotate=(-30, 30)),
            iaa.ShearY((-40,40)),
            iaa.ShearX((-40,40)),
            
            
        ], random_order=True)

        self.debug_dataset = True;
        pass
    def initialize_augmentation(self, radiographs, masks, layer_names):
        o = QtCore.QDir.currentPath();
        radiograph_list = [];
        mask_list = [];
        if not os.path.exists(os.path.sep.join([o,'Aug-Tmp'])):
            os.makedirs(os.path.sep.join([o,'Aug-Tmp']));
        
        self.clear_augmentations();

        num_classes = len(layer_names);

        #For cost-sensitive learning
        #We should add one for the background layer
        layer_weight = np.zeros((len(layer_names),2), dtype=np.float);

        
        for i in range(len(radiographs)):
            #load both radiograph and mask
            radiograph_image = cv2.imread(radiographs[i],cv2.IMREAD_UNCHANGED);
            #print(radiographs[i]);
            mask_image = np.zeros(shape = (radiograph_image.shape[0],
             radiograph_image.shape[1], num_classes), dtype=np.uint8);

            #Open meta file and open relating masks
            df = pd.read_pickle(masks[i]);
            for k in range(len(layer_names)):
                #if the layer has been selected by user read data
                desc = df[layer_names[k]];
                mask_name = desc[2];
                layer = desc[0];
                mask_image_layer = cv2.imread(os.path.sep.join([Config.PROJECT_ROOT, 'labels', mask_name]),cv2.IMREAD_UNCHANGED);
                mask_image_layer = np.sum(mask_image_layer[:,:,:3], axis=2);
                marked_pixels = (mask_image_layer != 0);
                #set the corresponding class to one
                mask_image[marked_pixels == True,k] = 1;

                layer_weight[k][0] += np.sum(marked_pixels);
                layer_weight[k][1] += mask_image_layer.shape[0] * mask_image_layer.shape[1];

            # #count each layer classes in mask image and them
            # for j in range(len(layer_weight)):
            #     c = (mask_image == j).sum();
            #     layer_weight[j] = layer_weight[j] + c;


            segmap = SegmentationMapsOnImage(mask_image, shape=radiograph_image.shape);

            filename = os.path.basename(radiographs[i]);
            filename = filename[:filename.find('.')];

            #Add original image as the first in list
            rad_path = os.path.sep.join([o,'Aug-Tmp', f'{filename}-({int(0)}).png']);
            mask_path = os.path.sep.join([o,'Aug-Tmp', f'{filename}-m-({int(0)}).msk']);

            cv2.imwrite(rad_path,np.array(radiograph_image));
            pickle.dump(mask_image, open(mask_path, "wb"));

            # if self.debug_dataset == True:
            #     mask_path_debug = os.path.sep.join([o,'Aug-Tmp', f'{filename}-md-({int(0)}).png']);
            #     d = segmap.draw(size=segmap.shape[:2])[0];
            #     cv2.imwrite(mask_path_debug,d);

            radiograph_list.append(rad_path);
            mask_list.append(mask_path);

            for k in range(0):
                image_aug, segmap_aug = self.seq(image=radiograph_image, segmentation_maps=segmap);
                segmap_aug_arr = segmap_aug.arr;

                rad_path = os.path.sep.join([o,'Aug-Tmp', f'{filename}-({int(k+1)}).png']);
                mask_path = os.path.sep.join([o,'Aug-Tmp', f'{filename}-m-({int(k+1)}).png']);
                
                cv2.imwrite(rad_path,np.array(image_aug));
                cv2.imwrite(mask_path,np.array(segmap_aug_arr));

                radiograph_list.append(rad_path);
                mask_list.append(mask_path);

                if self.debug_dataset == True:
                    mask_path_debug = os.path.sep.join([o,'Aug-Tmp', f'{filename}-md-({int(k+1)}).png']);
                    d = segmap_aug.draw(size=image_aug.shape[:2])[0];
                    cv2.imwrite(mask_path_debug,d);
            #-------------------------------------------------------------------

        print(f"[TRAIN INFO] | Train size(after augmentation): {len(radiograph_list)}");
        #Shuffle list
        radiograph_list, mask_list = shuffle(radiograph_list, mask_list,random_state=10);
        return radiograph_list, mask_list, layer_weight;
    
    def clear_augmentations(self):
        o = QtCore.QDir.currentPath();
        aug_files = glob(os.path.sep.join([o,'Aug-Tmp'])+'/*');
        for f in aug_files:
            os.unlink(f);


class NetworkDataset(Dataset):
    def __init__(self, radiographs, masks, transform, layer_names = None, train = True):
        if train:
            logging.info("Train_");
            self.radiographs = radiographs;
            self.masks = masks;

            self.transform = transform;
        else:
            logging.info("Validation_");
            o = QtCore.QDir.currentPath();
            self.masks = [];
            self.radiographs = radiographs;
            self.transform = transform;
            
            if not os.path.exists(os.path.sep.join([o,'valid-masks'])):
                os.makedirs(os.path.sep.join([o,'valid-masks']));

            for m in range(len(masks)):
                dummy = cv2.imread(radiographs[m],cv2.IMREAD_UNCHANGED);
                filename = os.path.basename(masks[m]);
                filename = filename[:filename.find('.')];
                df = pd.read_pickle(masks[m]);
                num_classes = len(layer_names);

                mask_image = np.zeros(shape = (dummy.shape[0], dummy.shape[1], num_classes), dtype=np.uint8);
                for k in range(len(layer_names)):
                    desc = df[layer_names[k]];
                    mask_name = desc[2];
                    layer = desc[0];
                    
                    mask_image_layer = cv2.imread(os.path.sep.join([Config.PROJECT_ROOT, 'labels', mask_name]),cv2.IMREAD_UNCHANGED);
                    mask_image_layer = np.sum(mask_image_layer[:,:,:3], axis=2);

                    marked_pixels = (mask_image_layer != 0);

                    #set the corresponding class to one
                    mask_image[marked_pixels == True,k] = 1;

                mask_path = os.path.sep.join([o,'valid-masks', f'{filename}-m.msk']);
                self.masks.append(mask_path);
                pickle.dump(mask_image, open(mask_path, 'wb'));
            
    def __len__(self):
        logging.info(f"Data size: {len(self.radiographs)}");
        return len(self.radiographs);

    def __getitem__(self, index):
        logging.info("start of get item");

        radiograph_image_path = self.radiographs[index];
        
        radiograph_image = cv2.imread(radiograph_image_path,cv2.IMREAD_GRAYSCALE);
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        radiograph_image = clahe.apply(radiograph_image);
        radiograph_image = np.expand_dims(radiograph_image, axis=2);
        radiograph_image = np.repeat(radiograph_image, 3,axis=2);

        if self.masks is not None:
            mask_image_path = self.masks[index];
            mask_image = pickle.load(open(mask_image_path, "rb"));
            transformed = self.transform(image = radiograph_image, mask = mask_image);
            radiograph_image = transformed["image"];
            mask_image = transformed['mask'];
            #ri = radiograph_image.permute(1,2,0).cpu().detach().numpy()*255;
            #sns.distplot(ri.ravel(), label=f'Mean : {np.mean(ri)}, std: {np.std(ri)}');
            #plt.legend(loc='best');
            #plt.savefig('dist-after.png');
            #mask_image = transformed["mask"]/255;
            #mask_image = torch.unsqueeze(mask_image, 0);
            return radiograph_image, mask_image, index;

        transformed = self.transform(image = radiograph_image);
        radiograph_image = transformed["image"];
        return radiograph_image, index;
            
def analyze_dataset():
    radiograph_root = os.path.sep.join(["dataset","CXR_png"]);
    mask_root = os.path.sep.join(["dataset","masks"]);

    radiograph_list = os.listdir(radiograph_root);
    mask_list = os.listdir(mask_root);

    mask_images_names = [];
    radiograph_images_names = [];

    negative = 0;
    positive = 0;
    for m in radiograph_list:
        b = m.find('MCU');
        mask_name = m[0:m.find('.')] + "_mask.png" if b else m;
        if mask_name in mask_list:
            mask_images_names.append(mask_name);
            radiograph_images_names.append(m);
            sample_img = cv2.imread(os.path.sep.join([mask_root,mask_name]),cv2.IMREAD_GRAYSCALE);
            w,h = sample_img.shape;
            sample_img = sample_img.flatten();
            sample_img = (sample_img == 255);
            p = np.sum(sample_img);
            n = np.sum((sample_img == 0))
            positive += p / (704 * w*h);
            negative += n/ (704 * w*h);
    
    print(f"Positive portion:{positive}\tNegative portion:{negative}");
    negative_bias = positive;


    pass

if __name__ == "__main__":
    analyze_dataset();