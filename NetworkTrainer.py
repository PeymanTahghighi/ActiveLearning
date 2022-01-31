import pickle
from PyQt5 import QtCore
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox
from django import conf
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
from NetworkDataset import NetworkDataset, OfflineAugmentation, TrainValidSplit
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
from albumentations.pytorch import ToTensorV2
import copy
import PIL.ImageColor
import torchvision.transforms.functional as F
from ignite.contrib.handlers.tensorboard_logger import *
import Config
import logging
from torchmetrics import *
#import ptvsd
from StoppingStrategy import *


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
        self.model_load_status = False;
        pass

    #This function should be called once the program starts
    def __initialize(self,):
        self.disc_list = [];

        self.model = Unet(num_classes=3).to(Config.DEVICE);
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4);

        self.precision_estimator = Precision(num_classes=3, average='macro').to(Config.DEVICE);
        self.recall_estimator = Recall(num_classes=3, average='macro').to(Config.DEVICE);
        self.accuracy_esimator = Accuracy(num_classes=3,average='macro').to(Config.DEVICE);
        self.f1_esimator = F1(num_classes=3, average='macro').to(Config.DEVICE);

        self.l1_loss = nn.L1Loss().to(Config.DEVICE);

        self.scaler = torch.cuda.amp.grad_scaler.GradScaler()
        
        #Initialize transforms for training and validation
        self.train_transforms = A.Compose(
            [
                #A.PadIfNeeded(min_height = 512, min_width = 512),
                #A.RandomCrop(Config.IMAGE_SIZE, Config.IMAGE_SIZE, always_apply = False, p = 0.5),
                #A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                #A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=20, p=0.5),
                #A.HorizontalFlip(p=0.5),
                #A.RandomBrightnessContrast(p=0.5),
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
                ToTensorV2(),
            ],
            additional_targets={'mask': 'mask'}
        )

        self.valid_transforms = A.Compose(
                [
                #A.PadIfNeeded(min_height = 512, min_width = 512),
                #A.RandomCrop(Config.IMAGE_SIZE, Config.IMAGE_SIZE, always_apply = True),
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
                ToTensorV2()
                ]
        )
    
        self.train_valid_split = TrainValidSplit();
        self.offline_augmentation = OfflineAugmentation();
        pass

    #This function should be called everytime we want to train a new model
    def initialize_new_train(self, layer_names):
        
        self.writer = SummaryWriter(os.path.sep.join([Config.PROJECT_ROOT,'experiments']));

        train_radiograph, train_masks, valid_radiographs, valid_masks = \
        self.train_valid_split.get(os.path.sep.join([Config.PROJECT_ROOT,'images']), 
            os.path.sep.join([Config.PROJECT_ROOT,'labels']),0.2, layer_names);
        train_radiograph, train_masks, layer_weight = self.offline_augmentation.initialize_augmentation(train_radiograph, train_masks, layer_names);

        self.train_dataset = NetworkDataset(train_radiograph, train_masks, self.train_transforms, train = True);
        self.valid_dataset = NetworkDataset(valid_radiographs, valid_masks, self.valid_transforms, train = False, layer_names = layer_names);

        self.train_loader = DataLoader(self.train_dataset, 
        batch_size= Config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True);

        self.valid_loader = DataLoader(self.valid_dataset, 
        batch_size= Config.BATCH_SIZE, shuffle=False);

        #self.gen.set_num_classes(Config.NUM_CLASSES);

        #set weight tensor by calculating each class distribution
        total = np.sum(layer_weight);
        for i in range(len(layer_weight)):
            layer_weight[i] = total / layer_weight[i];

        #Normalize to have numbers between 0 and 1
        layer_weight = layer_weight / np.sqrt(np.sum(layer_weight **2));
        self.weight_tensor = torch.tensor(layer_weight,dtype=torch.float32);
        if Config.NUM_CLASSES > 2:
            self.bce = nn.CrossEntropyLoss().to(Config.DEVICE);
        else:
            self.ce = nn.BCELoss().to(Config.DEVICE);
        
        self.stopping_strategy = CombinedTrainValid(2,5);

        self.model.reset_weights();

        #Initialize the weights of generator and discriminator
        #self.disc.apply(self.initialize_weights)
        #self.gen.apply(self.initialize_weights)

        #Finally save train meta file.
        #It describes number of classes and each layer's name
        pickle.dump([Config.NUM_CLASSES, layer_names], open(os.path.sep.join([Config.PROJECT_ROOT,'ckpts','train.meta']),'wb'));
    
    def dice_loss(self, pred, mask, eps=1e-7):
        """Computes the Sørensen–Dice loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the dice loss so we
        return the negated dice loss.
        Args:
            true: a tensor of shape [B, 1, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.
        Returns:
            dice_loss: the Sørensen–Dice loss.
        """
        #num_classes = pred.shape[1]
        if Config.NUM_CLASSES == 2:
            mask = mask.permute(0,2,3,1).long();
            pred = pred.permute(0,2,3,1).long();
        else:
            mask = mask.permute(0,2,3,1).long();
            pred = pred.permute(0,2,3,1).long();

            mask = torch.nn.functional.one_hot(mask, num_classes = Config.NUM_CLASSES).squeeze(dim = 3);
            pred = torch.nn.functional.one_hot(pred, num_classes = Config.NUM_CLASSES).squeeze(dim = 3);

            #probas = F.softmax(logits, dim=1)
        #true_1_hot = true_1_hot.type(logits.type())
        dims = tuple(range(1, mask.ndimension()))
        intersection = torch.sum(mask * pred, dims);
        cardinality = torch.sum(mask + pred, dims)
        dice_loss = (2. * intersection / (cardinality + eps)).mean()
        return (1 - dice_loss)

    def __loss_func_gen(self, pred_mask, pred_mask_thresh, gt_mask, d_real_list, d_fake_list, ce):
        logging.info("Inside loss function");
        loss_l1 = 0;
        for i in range(len(d_fake_list)):
            loss_l1 += self.weight_tensor[i] * torch.mean(torch.abs(d_fake_list[i] - d_real_list[i]));
        
        loss_l1 /= len(d_fake_list);
        
        loss_dice = self.dice_loss(pred_mask_thresh, gt_mask.long());
        
        #loss_bce = bce(pred_mask, gt_mask.float());
        # gt_mask = gt_mask.long();
        # prednp = pred_mask.cpu().detach().numpy();
        # gtnp = gt_mask.cpu().detach().numpy();
        # loss_pos_np = -1*np.mean(gtnp * np.log(prednp + Config.EPSILON));
        # loss_neg_np = -1*np.mean((1-gtnp) * np.log((1-prednp+ Config.EPSILON)));
        
        if Config.NUM_CLASSES > 2:
            loss_seg = ce(pred_mask, gt_mask.squeeze(dim=1).long());
        else:
            loss_pos = -1*torch.mean(gt_mask*torch.log(pred_mask + Config.EPSILON))  * self.weight_tensor[1];
            loss_neg = -1*torch.mean((1-gt_mask)*torch.log(1-pred_mask + Config.EPSILON)) * self.weight_tensor[0];
            loss_seg = loss_pos + loss_neg;

        return 10.0*loss_seg + 10.0*loss_dice + 10.0*loss_l1;

    def __train_one_epoch(self, loader, model, optimizer):

        epoch_loss = 0;
        step = 0;
        update_step = 2;
        with tqdm(loader, unit="batch") as batch_data:
            for radiograph, mask, _ in batch_data:
                radiograph,mask = radiograph.to(Config.DEVICE), mask.to(Config.DEVICE)
                # radiograph,mask = radiograph.to(Config.DEVICE), mask.to(Config.DEVICE);
                # radiograph_np = radiograph.permute(0,2,3,1).cpu().detach().numpy();
                #mask_np = mask.permute(0,2,3,1).cpu().detach().numpy();
                # radiograph_np = radiograph_np*0.5+0.5;
                # plt.figure();
                # plt.imshow(radiograph_np[0]);
                # plt.waitforbuttonpress();

                # plt.figure();
                # plt.imshow(mask[0]*255);
                # plt.waitforbuttonpress();
                

                with torch.cuda.amp.autocast_mode.autocast():
                    pred,_ = model(radiograph);
                    loss_ce = self.bce(pred,mask.squeeze(dim=1).long());

                    loss = loss_ce;

                self.scaler.scale(loss).backward();
                epoch_loss += loss.item();
                step += 1;

                if step % update_step == 0:
                    self.scaler.step(optimizer);
                    self.scaler.update();
                    optimizer.zero_grad();

    def __eval_one_epoch(self, loader, model):
        epoch_loss = 0;
        total_pred_lbl = None;
        total_mask = None;
        first = True;
        count = 0;
        
        with torch.no_grad():
            with tqdm(loader, unit="batch") as epoch_data:
                for radiograph, mask, _ in epoch_data:
                    radiograph,mask = radiograph.to(Config.DEVICE), mask.to(Config.DEVICE);

                    pred,_ = model(radiograph);
                    loss = self.bce(pred,mask.squeeze(dim=1).long());

                    epoch_loss += loss.item();
                    
                    if first is True:
                        total_pred = pred;
                        total_mask = mask;
                        first = False;
                    else:
                        total_pred = torch.cat([total_pred, pred], dim=0);
                        total_mask = torch.cat([total_mask, mask], dim=0);

                    count += 1;
        total_pred_lbl =  torch.argmax(torch.softmax(total_pred,dim=1),dim=1);
        total_mask = total_mask;
        prec = self.precision_estimator(total_pred_lbl.flatten(), total_mask.flatten().long());
        rec = self.recall_estimator(total_pred_lbl.flatten(), total_mask.flatten().long());
        acc = self.accuracy_esimator(total_pred_lbl.flatten(), total_mask.flatten().long());
        f1 = self.f1_esimator(total_pred_lbl.flatten(), total_mask.flatten().long());
        return epoch_loss / count, acc, prec, rec, f1;


    def start_train_slot(self, layers_names):
        logging.info("Start training...");
        self.initialize_new_train(layers_names);

        best = 100;
        e = 0;
        best_model = None;

        while(True):
            self.model.train();
            self.__train_one_epoch(self.train_loader,self.model, self.optimizer);

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

    def load_model(self):
        lstdir = glob(os.path.sep.join([Config.PROJECT_ROOT,'ckpts']) + '/*');
        #Find checkpoint file based on extension
        found = False;
        for c in lstdir:
            file_name, ext = os.path.splitext(c);
            if ext == '.pt':
                found = True;
                #Load train meta to read layer names and number of classes
                self.train_meta = pickle.load(open(os.path.sep.join([Config.PROJECT_ROOT,'ckpts', 'train.meta']),'rb'));
                Config.NUM_CLASSES = self.train_meta[0];
                print(Config.NUM_CLASSES);
                #self.gen.set_num_classes(Config.NUM_CLASSES);
                load_checkpoint(c, self.model);
                #self.model.load_state_dict(checkpoint["state_dict"])

        self.model_loaded_finished.emit(found);
        self.model_load_status = found;

    '''
        Predict of unlabeled data and update the second entry in  dictionary to 1.
    '''
    def predict(self, lbl, dc):
        #ptvsd.debug_this_thread();
        #Because predicting is totally based on the initialization of the model,
        # if we haven't loaded a model yet or the loading wasn't successfull
        # we should not do anything and return immediately.
        if self.model_load_status:
            self.model.eval();
            with torch.no_grad():
                radiograph_image = cv2.imread(os.path.sep.join([Config.PROJECT_ROOT, 'images', lbl]),cv2.IMREAD_GRAYSCALE);
                #radiograph_image = cv2.imread(radiograph_image_path,cv2.IMREAD_GRAYSCALE);
                clahe = cv2.createCLAHE(7,(11,11));
                radiograph_image = clahe.apply(radiograph_image);
                radiograph_image = np.expand_dims(radiograph_image, axis=2);
                radiograph_image = np.repeat(radiograph_image, 3,axis=2);
                
                w,h,_ = radiograph_image.shape;
                transformed = self.valid_transforms(image = radiograph_image);
                radiograph_image = transformed["image"];
                radiograph_image = radiograph_image.to(Config.DEVICE);
                #radiograph_image = torch.unsqueeze(radiograph_image,0);
                #radiograph_image = radiograph_image.to(Config.DEVICE);
                p,_ = self.model(radiograph_image.unsqueeze(dim=0));
                mask_list = [];
                if Config.NUM_CLASSES > 2:
                    num_classes = p.size()[1];
                    p = p.permute(0,2,3,1).cpu().detach().numpy()[0];
                    p = np.argmax(p, axis=2);
                    
                    #Convert each class to a predefined color
                    for i in range(1,num_classes):
                        mask = np.zeros(shape=(Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3),dtype=np.uint8);
                        tmp = (p==i);
                        mask[(tmp)] = Config.PREDEFINED_COLORS[i-1];
                        mask = cv2.resize(mask,(h,w), interpolation=cv2.INTER_NEAREST);
                        mask_list.append(mask);
                else:
                    p = (p > 0.5).long();
                    p = p.permute(0,2,3,1).cpu().detach().numpy()[0];
                    mask = np.zeros(shape=(Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3),dtype=np.uint8);
                    tmp = (p==1).squeeze();
                    a = PIL.ImageColor.getrgb(Config.PREDEFINED_COLORS[0]);
                    mask[(tmp)] = PIL.ImageColor.getrgb(Config.PREDEFINED_COLORS[0]);
                    mask = cv2.resize(mask,(h,w), interpolation=cv2.INTER_NEAREST);
                    mask_list.append(mask);
                self.predict_finished_signal.emit(mask_list, self.train_meta[1]);


                # p = p[0]*255;
                # p = np.array(p, dtype=np.uint8);
                

                # kernel = np.ones((9,9), dtype=np.uint8);
                # opening = cv2.morphologyEx(p, cv2.MORPH_OPEN, kernel);
                # close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel);

                #save image to predicition directory
                #cv2.imwrite(os.path.sep.join(['predictions',os.path.basename(lbl)]), mask);
