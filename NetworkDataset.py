from PyQt5 import QtCore
from gevent import config
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
from torch.utils.data.sampler import Sampler
from utils import JSD
import random

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

        print(f"[TRAIN INFO] | Train size: {len(selected_indices)} \tValid size: {dataset_size - len(selected_indices)}");

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
        #For cost-sensitive learning
        #We should add one for the background layer
        layer_weight = np.zeros((len(layer_names),2), dtype=np.float);

        
        for i in range(len(radiographs)):
            #load both radiograph and mask
            radiograph_image = cv2.imread(radiographs[i],cv2.IMREAD_UNCHANGED);
            #print(radiographs[i]);
            

            #Open meta file and open relating masks
            df = pd.read_pickle(masks[i]);
            if Config.MUTUAL_EXCLUSION is False:
                mask_image = np.zeros(shape = (radiograph_image.shape[0],
                radiograph_image.shape[1], Config.NUM_CLASSES), dtype=np.uint8);
                for k in range(Config.NUM_CLASSES):
                    #if the layer has been selected by user read data
                    desc = df[layer_names[k-1]];
                    mask_name = desc[2];
                    layer = desc[0];
                    mask_image_layer = cv2.imread(os.path.sep.join([Config.PROJECT_ROOT, 'labels', mask_name]),cv2.IMREAD_UNCHANGED);
                    mask_image_layer = np.sum(mask_image_layer[:,:,:3], axis=2);
                    marked_pixels = (mask_image_layer != 0);
                    #set the corresponding class to one
                    mask_image[marked_pixels == True,k] = 1;

                    layer_weight[k-1][0] += np.sum(marked_pixels);
                    layer_weight[k-1][1] += mask_image_layer.shape[0] * mask_image_layer.shape[1];
            else:
                mask_image = np.zeros(shape = (radiograph_image.shape[0],
                radiograph_image.shape[1]), dtype=np.uint8);
                for k in range(1,Config.NUM_CLASSES):
                    #if the layer has been selected by user read data
                    desc = df[layer_names[k-1]];
                    mask_name = desc[2];
                    layer = desc[0];
                    mask_image_layer = cv2.imread(os.path.sep.join([Config.PROJECT_ROOT, 'labels', mask_name]),cv2.IMREAD_UNCHANGED);
                    mask_image_layer = np.sum(mask_image_layer[:,:,:3], axis=2);
                    marked_pixels = (mask_image_layer != 0);
                    #set the corresponding class to one
                    mask_image[marked_pixels == True] = k;

                    layer_weight[k-1][0] += np.sum(marked_pixels);
                    layer_weight[k-1][1] += mask_image_layer.shape[0] * mask_image_layer.shape[1];

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
        self.__masks = [];

        #For cost-sensitive learning
        #We should add one for the background layer
        layer_weight = np.zeros((len(layer_names),2), dtype=np.float);
        
        for i in range(len(radiographs)):
            #load both radiograph and mask
            radiograph_image = cv2.imread(radiographs[i],cv2.IMREAD_UNCHANGED);

            #Open meta file and open relating masks
            df = pd.read_pickle(masks[i]);
            if Config.MUTUAL_EXCLUSION is False:
                mask_image = np.zeros(shape = (radiograph_image.shape[0],
                radiograph_image.shape[1], Config.NUM_CLASSES), dtype=np.uint8);
                for k in range(Config.NUM_CLASSES):
                    #if the layer has been selected by user read data
                    desc = df[layer_names[k-1]];
                    mask_name = desc[2];
                    mask_image_layer = cv2.imread(os.path.sep.join([Config.PROJECT_ROOT, 'labels', mask_name]),cv2.IMREAD_UNCHANGED);
                    mask_image_layer = np.sum(mask_image_layer[:,:,:3], axis=2);
                    marked_pixels = (mask_image_layer != 0);
                    #set the corresponding class to one
                    mask_image[marked_pixels == True,k] = 1;

                    layer_weight[k-1][0] += np.sum(marked_pixels);
                    layer_weight[k-1][1] += mask_image_layer.shape[0] * mask_image_layer.shape[1];
            else:
                mask_image = np.zeros(shape = (radiograph_image.shape[0],
                radiograph_image.shape[1]), dtype=np.uint8);
                for k in range(1,Config.NUM_CLASSES):
                    #if the layer has been selected by user read data
                    desc = df[layer_names[k-1]];
                    mask_name = desc[2];
                    layer = desc[0];
                    mask_image_layer = cv2.imread(os.path.sep.join([Config.PROJECT_ROOT, 'labels', mask_name]),cv2.IMREAD_UNCHANGED);
                    mask_image_layer = np.sum(mask_image_layer[:,:,:3], axis=2);
                    marked_pixels = (mask_image_layer != 0);
                    #set the corresponding class to one
                    mask_image[marked_pixels == True] = k;

                    layer_weight[k-1][0] += np.sum(marked_pixels);
                    layer_weight[k-1][1] += mask_image_layer.shape[0] * mask_image_layer.shape[1];

            #segmap = SegmentationMapsOnImage(mask_image, shape=radiograph_image.shape);

            filename = os.path.basename(radiographs[i]);
            filename = filename[:filename.find('.')];

            #Add original image as the first in list
            #rad_path = os.path.sep.join(['masks', f'{filename}-({int(0)}).png']);
            mask_path = os.path.sep.join(['masks', f'{filename}-m-({int(0)}).msk']);

            #cv2.imwrite(rad_path,np.array(radiograph_image));
            pickle.dump(mask_image, open(mask_path, "wb"));

            # if self.debug_dataset == True:
            #     mask_path_debug = os.path.sep.join([o,'Aug-Tmp', f'{filename}-md-({int(0)}).png']);
            #     d = segmap.draw(size=segmap.shape[:2])[0];
            #     cv2.imwrite(mask_path_debug,d);

            self.__masks.append(mask_path);

        self.__radiographs = radiographs;
        self.__transform = transform;

    def __len__(self):
        return len(self.__radiographs);

    def __getitem__(self, index):
        radiograph_image_path = self.__radiographs[index];
        
        radiograph_image = cv2.imread(radiograph_image_path,cv2.IMREAD_GRAYSCALE);
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        radiograph_image = clahe.apply(radiograph_image);
        radiograph_image = np.expand_dims(radiograph_image, axis=2);
        radiograph_image = np.repeat(radiograph_image, 3,axis=2);

        #if self.masks is not None:
        mask_image_path = self.__masks[index];
        mask_image = pickle.load(open(mask_image_path, "rb"));
        transformed = self.__transform(image = radiograph_image, mask = mask_image);
        radiograph_image = transformed["image"];
        mask_image = transformed['mask'];
        
        # _,w,h = radiograph_image.shape;

        # pad_width = 32 - w%32;
        # pad_height = 32 - h%32;

        # padded_radiograph = torch.zeros((3, w + pad_width, h + pad_height), dtype=torch.float32);

        # if len(mask_image.shape) > 2:
        #     msk_size = mask_image.shape[-1];
        #     padded_seg = torch.zeros((w + pad_width, h + pad_height, msk_size), dtype=torch.float32);
        # else:
        #     padded_seg = torch.zeros((w + pad_width, h + pad_height), dtype=torch.float32);

        # padded_radiograph[:, :w, :h] = radiograph_image;
        # padded_seg[:w, :h] = mask_image;



        #ri = radiograph_image.permute(1,2,0).cpu().detach().numpy()*255;
        #sns.distplot(ri.ravel(), label=f'Mean : {np.mean(ri)}, std: {np.std(ri)}');
        #plt.legend(loc='best');
        #plt.savefig('dist-after.png');
        #mask_image = transformed["mask"]/255;
        #mask_image = torch.unsqueeze(mask_image, 0);
        return radiograph_image, mask_image;

        # transformed = self.transform(image = radiograph_image);
        # radiograph_image = transformed["image"];
        # return radiograph_image, index;
    
    def image_aspect_ratio(self, index):
        radiograph_image = cv2.imread(self.__radiographs[index], cv2.IMREAD_GRAYSCALE);
        return float(radiograph_image.shape[1]) / float(radiograph_image.shape[0])


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

def collater(data):

    imgs = [s[0] for s in data]
    segs = [s[1] for s in data]
        
    widths = [int(s.shape[1]) for s in imgs]
    heights = [int(s.shape[2]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, 3, max_width, max_height)

    msk_size = 0;
    if len(segs[0].shape) > 2:
        msk_size = segs[0].shape[-1];
        padded_segs = torch.zeros(batch_size, max_width, max_height, msk_size)
    else:
        padded_segs = torch.zeros(batch_size, max_width, max_height)

    for i in range(batch_size):
        img = imgs[i]
        seg = segs[i]
        padded_imgs[i, :,  :int(img.shape[1]), :int(img.shape[2])] = img

        if msk_size == 0:
            padded_segs[i, :int(seg.shape[0]), :int(seg.shape[1])] = seg;
        else:
            padded_segs[i, :int(seg.shape[0]), :int(seg.shape[1]), :] = seg;


    return padded_imgs, padded_segs;