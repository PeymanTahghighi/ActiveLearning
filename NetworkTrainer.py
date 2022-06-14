import pickle
from PyQt5 import QtCore
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox
from ignite.handlers.checkpoint import Checkpoint
from ignite.metrics.multilabel_confusion_matrix import MultiLabelConfusionMatrix
from ignite.metrics.confusion_matrix import ConfusionMatrix
import ignite.utils as igut
from sklearn.utils import shuffle
from PyQt5.QtCore import QObject
from numpy.core.fromnumeric import mean
from numpy.lib.npyio import load
from torch.nn.modules.loss import L1Loss
from torch.utils import data
from utils import load_checkpoint, save_checkpoint, save_samples
import torch
import torch.nn as nn
import torch.optim as optim
import Config
from NetworkDataset import AspectRatioBasedSampler, NetworkDataset, OfflineAugmentation, TrainValidSplit
from Network import *
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.transforms import transforms
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from NetworkEvaluation import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_curve, precision_recall_curve
from PyQt5.QtCore import pyqtSlot, QObject, pyqtSignal
from torchvision.utils import save_image
from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import RunningAverage
from ignite.handlers import ModelCheckpoint, Timer, EarlyStopping, global_step_from_engine
import os
from PIL import Image
from glob import glob
from torchvision.utils import save_image
import albumentations as A
import PIL.ImageColor
import torchvision.transforms.functional as F
from ignite.contrib.handlers.tensorboard_logger import *
import Config
import logging
from torchmetrics import *
import ptvsd
from StoppingStrategy import *
from Loss import dice_loss, focal_loss, tversky_loss
from utils import JSD
from NetworkDataset import collater

class NetworkTrainer(QObject):
    train_finsihed_signal = pyqtSignal();
    predict_finished_signal = pyqtSignal(list, list);
    update_train_info_iter = pyqtSignal(float);
    update_valid_info_iter = pyqtSignal(float);
    update_train_info_epoch_train = pyqtSignal(list,list,float,int);
    update_train_info_epoch_valid = pyqtSignal(list,list);
    augmentation_finished_signal = pyqtSignal();
    model_loaded_finished = pyqtSignal(bool);

    def __init__(self):
        super().__init__();
        self.__initialize();
        #To check if we've successfuly opened a model from disk or not.
        self.__model_load_status = False;
        pass

    #This function should be called once the program starts
    def __initialize(self,):

        self.model = Unet().to(Config.DEVICE);
        self.l1_loss = nn.L1Loss().to(Config.DEVICE);

        self.scaler = torch.cuda.amp.grad_scaler.GradScaler()
    
        self.train_valid_split = TrainValidSplit();
        self.offline_augmentation = OfflineAugmentation();
        pass
    
    def get_model(self,):
        if self.__model_load_status is False:
            self.load_model();

        return self.model, self.__model_load_status;
    
    def initialize_new_train(self, layer_names):
        
        #set model_load_satus to false so next time we are going to use the model
        #we are forced to load the newly trained model
        self.__model_load_status = False;
        
        self.model.set_num_classes(Config.NUM_CLASSES);

        self.optimizer = optim.RMSprop(self.model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4);

        self.precision_estimator = Precision(num_classes = 1 if Config.MUTUAL_EXCLUSION == False else Config.NUM_CLASSES).to(Config.DEVICE);
        self.recall_estimator = Recall(num_classes = 1 if Config.MUTUAL_EXCLUSION == False else Config.NUM_CLASSES).to(Config.DEVICE);
        self.accuracy_esimator = Accuracy(num_classes = 1 if Config.MUTUAL_EXCLUSION == False else Config.NUM_CLASSES).to(Config.DEVICE);
        self.f1_esimator = F1Score(num_classes = 1 if Config.MUTUAL_EXCLUSION == False else Config.NUM_CLASSES).to(Config.DEVICE);

        self.writer = SummaryWriter(os.path.sep.join([Config.PROJECT_ROOT,'experiments']));

        train_radiograph, train_masks, valid_radiographs, valid_masks = \
        self.train_valid_split.get(os.path.sep.join([Config.PROJECT_ROOT,'images']), 
            os.path.sep.join([Config.PROJECT_ROOT,'labels']),0.1, layer_names);
        train_radiograph, train_masks, layer_weight = self.offline_augmentation.initialize_augmentation(train_radiograph, train_masks, layer_names);

        self.train_dataset = NetworkDataset(train_radiograph, train_masks, Config.train_transforms, train = True);
        self.valid_dataset = NetworkDataset(valid_radiographs, valid_masks, Config.valid_transforms, train = False, layer_names = layer_names);

        self.sampler_train = AspectRatioBasedSampler(self.train_dataset, batch_size=Config.BATCH_SIZE, drop_last=False);
        #self.sampler_valid = AspectRatioBasedSampler(self.valid_dataset, batch_size=Config.BATCH_SIZE, drop_last=False);

        self.train_loader = DataLoader(self.train_dataset, collate_fn=collater, num_workers=0, batch_sampler=self.sampler_train);

        self.valid_loader = DataLoader(self.valid_dataset, batch_size=1, );

        #self.gen.set_num_classes(Config.NUM_CLASSES);

        #set weight tensor by calculating each class distribution
        # total = np.sum(layer_weight);
        # for i in range(len(layer_weight)):
        #     layer_weight[i] = total / layer_weight[i];
        # weight_tensor = np.zeros((Config.NUM_CLASSES), dtype=np.float32);
        # for n in range(Config.NUM_CLASSES):

        #     weight_tensor[n] = (layer_weight[n][1] / layer_weight[n][0]);

        # #Normalize to have numbers between 0 and 1
        # #layer_weight = layer_weight / np.sqrt(np.sum(layer_weight **2));
        # weight_tensor = torch.tensor(weight_tensor,dtype=torch.float32);
        self.bce = nn.BCEWithLogitsLoss().to(Config.DEVICE);

        self.stopping_strategy = CombinedTrainValid(1,5);

        self.model.reset_weights();

        #Initialize the weights of generator and discriminator
        #self.disc.apply(self.initialize_weights)
        #self.gen.apply(self.initialize_weights)

        #Finally save train meta file.
        #It describes number of classes and each layer's name
        pickle.dump([Config.NUM_CLASSES, layer_names], open(os.path.sep.join([Config.PROJECT_ROOT,'ckpts','train.meta']),'wb'));
    
    
    def __loss_func(self, output, gt):
        total_loss = 0;
        # for i in range(Config.NUM_CLASSES):
        #     curr_out = output[:,:,:,i];
        #     curr_gt = gt[:,:,:,i];

        #     # curr_gt_np = curr_gt[0].detach().cpu().numpy();
        #     # cv2.imshow("t", curr_gt_np);
        #     # cv2.waitKey();

        #output = torch.sigmoid(output);
        #output = torch.where(output > 0.5, 1.0, 0.0);
        f_loss = focal_loss(output, gt,  arange_logits=True, mutual_exclusion=True);
        t_loss = tversky_loss(output, gt, sigmoid=True, arange_logits=True, mutual_exclusion=True)
        #bce_loss = self.bce(output.permute(0,2,3,1), gt.float());
            #total_loss += bce_loss;
        
        return  t_loss + f_loss;
        


    def __train_one_epoch(self, loader, model, optimizer):

        epoch_loss = 0;
        step = 0;
        update_step = 1;
        with tqdm(loader, unit="batch") as batch_data:
            for radiograph, mask in batch_data:
                radiograph, mask = radiograph.to(Config.DEVICE), mask.to(Config.DEVICE)
                mask.require_grad = True;
                radiograph,mask = radiograph.to(Config.DEVICE), mask.to(Config.DEVICE);
                radiograph_np = radiograph.permute(0,2,3,1).cpu().detach().numpy();
                radiograph_np = radiograph_np[0][:,:,1];
                radiograph_np *= 0.229;
                radiograph_np += 0.485;
                radiograph_np *= 255;
                
                #cv2.imshow('radiograph', radiograph_np.astype("uint8"));
                #cv2.waitKey();
                # mask_np = mask.cpu().detach().numpy();
                # mask_np = mask_np[0];
                
                # radiograph_np = radiograph_np*0.5+0.5;
                # plt.figure();
                # plt.imshow(radiograph_np[0]);
                # plt.waitforbuttonpress();

                # cv2.imshow('mask', mask_np.astype("uint8")*255);
                # cv2.waitKey();

                # plt.figure();
                # plt.imshow(mask[0]*255);
                # plt.waitforbuttonpress();
                 

                with torch.cuda.amp.autocast_mode.autocast():
                    pred,_ = model(radiograph);
                    #sigmoid = nn.Sigmoid();
                    #pred = sigmoid(pred).permute(0,2,3,1);
                    loss = self.__loss_func(pred, mask);

                self.scaler.scale(loss).backward();
                epoch_loss += loss.item();
                step += 1;

                if step % update_step == 0:
                    self.scaler.step(optimizer);
                    self.scaler.update();
                    optimizer.zero_grad();

    def __eval_one_epoch(self, loader, model):
        epoch_loss = 0;
        total_prec = [];
        total_rec = [];
        total_f1 = [];
        total_acc = [];
        count = 0;
        
        with torch.no_grad():
            with tqdm(loader, unit="batch") as epoch_data:
                for radiograph, mask in epoch_data:
                    radiograph,mask = radiograph.to(Config.DEVICE), mask.to(Config.DEVICE);

                    pred,_ = model(radiograph);
                    loss = self.__loss_func(pred, mask);

                    epoch_loss += loss.item();
                    
                    if Config.MUTUAL_EXCLUSION is False:
                        pred = (torch.sigmoid(pred));
                        prec = self.precision_estimator(pred.flatten(), mask.flatten().long());
                        rec = self.recall_estimator(pred.flatten(), mask.flatten().long());
                        acc = self.accuracy_esimator(pred.flatten(), mask.flatten().long());
                        f1 = self.f1_esimator(pred.flatten(), mask.flatten().long());
                    else:
                        pred = (torch.softmax(pred, dim = 1)).permute(0,2,3,1);
                        prec = self.precision_estimator(pred, mask.long());
                        rec = self.recall_estimator(pred, mask.long());
                        acc = self.accuracy_esimator(pred, mask.long());
                        f1 = self.f1_esimator(pred, mask.long());
                    

                    total_prec.append(prec.item());
                    total_rec.append(rec.item());
                    total_f1.append(f1.item());
                    total_acc.append(acc.item());

                    count += 1;


        return epoch_loss / count, np.mean(total_acc), np.mean(total_prec), np.mean(total_rec), np.mean(total_f1);


    def start_train_slot(self, layers_names):
        ptvsd.debug_this_thread();
        logging.info("Start training...");
        self.initialize_new_train(layers_names);

        best = 100;
        e = 1;
        best_model = None;

        while(True):
            self.model.train();
            #self.__train_one_epoch(self.train_loader,self.model, self.optimizer);

            self.model.eval();
            train_loss, train_acc, train_precision, train_recall, train_f1 = self.__eval_one_epoch(self.train_loader, self.model);

            valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = self.__eval_one_epoch(self.valid_loader, self.model);

            print(f"Epoch {e}\tLoss: {train_loss}\tPrecision: {train_precision}\tRecall: {train_recall}\tAccuracy: {train_acc}\tF1: {train_f1}");
            print(f"Valid \tLoss: {valid_loss}\tPrecision: {valid_precision}\tRecall: {valid_recall}\tAccuracy: {valid_acc}\tF1: {valid_f1}");

            self.writer.add_scalar('training/loss', float(train_loss),e);
            self.writer.add_scalar('training/precision', float(train_precision),e);
            self.writer.add_scalar('training/recall', float(train_recall),e);
            self.writer.add_scalar('training/accuracy', float(train_acc),e);
            self.writer.add_scalar('training/f1', float(train_f1),e);

            self.writer.add_scalar('validation/loss', float(valid_loss),e);
            self.writer.add_scalar('validation/precision', float(valid_precision),e);
            self.writer.add_scalar('validation/recall', float(valid_recall),e);
            self.writer.add_scalar('validation/accuracy', float(valid_acc),e);
            self.writer.add_scalar('validation/f1', float(valid_f1),e);

            if(valid_loss < best):
                print("New best model found!");
                save_checkpoint(self.model, e);
                best = valid_loss;
                best_model = deepcopy(self.model.state_dict());
                save_samples(self.model, self.valid_loader, e, 'evaluation');

            if self.stopping_strategy(valid_loss, train_loss) is False:
                break;
            e += 1;

    def __load_model(self):
        
        lstdir = glob(os.path.sep.join([Config.PROJECT_ROOT,'ckpts']) + '/*');
        #Find checkpoint file based on extension
        found = False;
        for c in lstdir:
            file_name, ext = os.path.splitext(c);
            if ext == '.pt':
                #Load train meta to read layer names and number of classes
                self.train_meta = pickle.load(open(os.path.sep.join([Config.PROJECT_ROOT,'ckpts', 'train.meta']),'rb'));
                Config.NUM_CLASSES = self.train_meta[0];
                self.model.set_num_classes(Config.NUM_CLASSES);
                load_checkpoint(c, self.model);
                #self.model.load_state_dict(checkpoint["state_dict"])
                found = True;
        self.__model_load_status = found;
        return found;
    

    def load_model(self):
        return self.__load_model();

    def load_model_for_predicition(self):
        found = self.__load_model();
        self.model_loaded_finished.emit(found);

    '''
        Predict of unlabeled data and update the second entry in  dictionary to 1.
    '''
    def predict(self, lbl, dc):
        #Because predicting is totally based on the initialization of the model,
        # if we haven't loaded a model yet or the loading wasn't successfull
        # we should not do anything and return immediately.

        #ptvsd.debug_this_thread();
        if self.__model_load_status:
            self.model.eval();
            with torch.no_grad():
                radiograph_image = cv2.imread(os.path.sep.join([Config.PROJECT_ROOT, 'images', lbl]),cv2.IMREAD_GRAYSCALE);
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8));
                radiograph_image = clahe.apply(radiograph_image);
                radiograph_image = np.expand_dims(radiograph_image, axis=2);
                radiograph_image = np.repeat(radiograph_image, 3,axis=2);
                
                w,h,_ = radiograph_image.shape;
                transformed = Config.valid_transforms(image = radiograph_image);
                radiograph_image = transformed["image"];
                radiograph_image = radiograph_image.to(Config.DEVICE);
                #radiograph_image = torch.unsqueeze(radiograph_image,0);
                #radiograph_image = radiograph_image.to(Config.DEVICE);
                p,_ = self.model(radiograph_image.unsqueeze(dim=0));
                mask_list = [];

                p = torch.sigmoid(p);
                num_classes = p.size()[1];
                p = p.permute(0,2,3,1).cpu().detach().numpy()[0];
                #threshold to get the label
                p = p > 0.5;
                
                #Convert each class to a predefined color
                for i in range(num_classes):
                    mask = np.zeros(shape=(Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3),dtype=np.uint8);
                    mask_for_class = p[:,:,i];
                    tmp = (mask_for_class==1);
                    mask[(tmp)] = Config.PREDEFINED_COLORS[i];
                    mask = cv2.resize(mask,(h,w), interpolation=cv2.INTER_NEAREST);
                    mask_list.append(mask);
                # else:
                #     p = torch.sigmoid(p);
                #     p = (p > 0.5).long();
                #     p = p.permute(0,2,3,1).cpu().detach().numpy()[0];
                #     mask = np.zeros(shape=(Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3),dtype=np.uint8);
                #     tmp = (p==1).squeeze();
                #     a = PIL.ImageColor.getrgb(Config.PREDEFINED_COLORS[0]);
                #     mask[(tmp)] = PIL.ImageColor.getrgb(Config.PREDEFINED_COLORS[0]);
                #     mask = cv2.resize(mask,(h,w), interpolation=cv2.INTER_NEAREST);
                #     mask_list.append(mask);
                self.predict_finished_signal.emit(mask_list, self.train_meta[1]);