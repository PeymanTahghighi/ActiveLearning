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
from utils import load_checkpoint, save_checkpoint, save_some_examples
import torch
import torch.nn as nn
import torch.optim as optim
import Config
from NetworkDataset import NetworkDataset, OfflineAugmentation, TrainValidSplit
from Network import Generator, Discriminator
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
#import ptvsd


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

        self.gen = Generator(1, 128).to(Config.DEVICE);
        self.optGen = optim.RMSprop(self.gen.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4);

        self.l1_loss = nn.L1Loss().to(Config.DEVICE);
        
        #Initialize transforms for training and validation
        self.train_transforms = A.Compose(
            [
                #A.PadIfNeeded(min_height = 512, min_width = 512),
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                #A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=20, p=0.5),
                #A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=(0.5), std=(0.5)),
                ToTensorV2(),
            ],
            additional_targets={'mask': 'mask'}
        )

        self.valid_transforms = A.Compose(
                [
                #A.PadIfNeeded(min_height = 512, min_width = 512),
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.Normalize(mean=(0.5), std=(0.5)), 
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

        self.gen.set_num_classes(Config.NUM_CLASSES);

        #set weight tensor by calculating each class distribution
        total = np.sum(layer_weight);
        for i in range(len(layer_weight)):
            layer_weight[i] = total / layer_weight[i];

        #Normalize to have numbers between 0 and 1
        layer_weight = layer_weight / np.sqrt(np.sum(layer_weight **2));
        self.weight_tensor = torch.tensor(layer_weight,dtype=torch.float32);
        if Config.NUM_CLASSES > 2:
            self.ce = nn.NLLLoss(self.weight_tensor).to(Config.DEVICE);
        else:
            self.ce = nn.BCELoss().to(Config.DEVICE);
        

        #Initialize each descriminator
        self.disc_list.clear();
        for i in range(Config.NUM_CLASSES-1):
            disc = Discriminator().to(Config.DEVICE);
            opt_disc = optim.RMSprop(disc.parameters(), lr = Config.LEARNING_RATE, weight_decay=1e-4);
            self.disc_list.append([disc,opt_disc]);

        #Initialize the weights of generator and discriminator
        #self.disc.apply(self.initialize_weights)
        #self.gen.apply(self.initialize_weights)

        #Finally save train meta file.
        #It describes number of classes and each layer's name
        pickle.dump([Config.NUM_CLASSES, layer_names], open(os.path.sep.join([Config.PROJECT_ROOT,'ckpts','train.meta']),'wb'));

    def visualize_augmentations(self, dataset, idx=0, samples=5):
        dataset = copy.deepcopy(dataset)
        dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
        figure, ax = plt.subplots(nrows=samples, ncols=4, figsize=(10, 24))
        for i in range(samples):
            image, mask = dataset[idx]
            ax[i, 0].imshow(image, cmap='gray')
            ax[i, 1].imshow(mask, interpolation="nearest", cmap='gray');
            ax[i, 0].set_axis_off()
            ax[i, 1].set_axis_off()
        plt.tight_layout()
        plt.show()
        plt.waitforbuttonpress();
    
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

    def initialize_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            #torch.nn.init.xavier_uniform_(m.weight);
            m.weight.data.normal_(0.0, 1.0)
        # elif classname.find('BatchNorm') != -1:
        #     m.weight.data.normal_(1.0, 0.02)
        #     m.bias.data.fill_(0)

    def __train_step(self, engine, batch):
        
        #with torch.autograd.detect_anomaly():
        radiograph, mask = batch;
        radiograph,mask = radiograph.to(Config.DEVICE), mask.to(Config.DEVICE)

        #radiograph = (radiograph + 1)/2.0;

        epoch_disc_loss = 0;
        epoch_gen_loss = 0;

        y_fake = self.gen(radiograph);
        # y_fake_sm = torch.softmax(y_fake, dim=1);
        # y_fake_log = torch.log2(y_fake_sm);
        
        if Config.NUM_CLASSES > 2:
            y_fake_thresh_max = torch.argmax(y_fake, dim=1).unsqueeze(dim=1);
        else:
            y_fake_thresh_max = y_fake > 0.5;
        
        #n = y_fake_thresh_max.detach().numpy();
        y_fake_list = [];
        d_real_list = [];
        d_fake_list = [];

        #Train discriminators
        for i in range(1,Config.NUM_CLASSES):
            mask_tmp = mask.clone();
            y_fake_thresh_max_tmp = y_fake_thresh_max.clone();
            mask_tmp[mask_tmp != i] = 0;
            mask_tmp[mask_tmp == i] = 1;
            y_fake_thresh_max_tmp[y_fake_thresh_max_tmp != i] = 0;
            y_fake_thresh_max_tmp[y_fake_thresh_max_tmp == i] = 1;
        
            d_real = self.disc_list[i-1][0](radiograph,mask_tmp);

            d_fake = self.disc_list[i-1][0](radiograph,y_fake_thresh_max_tmp.detach());

            y_fake_list.append(y_fake_thresh_max_tmp);
            d_real_list.append(d_real.detach());

        
            loss_disc = -torch.mean(torch.abs(d_real - d_fake));

            self.disc_list[i-1][1].zero_grad();
            loss_disc.backward();
            self.disc_list[i-1][1].step();

            for p in self.disc_list[i-1][0].parameters():
                p.data.clamp_(-0.05, 0.05)
            
            epoch_disc_loss += loss_disc;
        #-----------------------------------------------------

        #Train generator
        for i in range(Config.NUM_CLASSES-1):
            #n = y_fake_list[i].detach().numpy();
            d_fake = self.disc_list[i][0](radiograph, y_fake_list[i]);
            d_fake_list.append(d_fake);

        loss_gen = self.__loss_func_gen(y_fake, y_fake_thresh_max, mask, 
            d_real_list, d_fake_list, self.ce);

        epoch_gen_loss = loss_gen.item();

        self.optGen.zero_grad();
        loss_gen.backward();

       # nn.utils.clip_grad_value_(self.gen.parameters(), clip_value=1.0)

        self.optGen.step();
        #------------------------------------------------------
        #self.apt.koft();
        return {"error_d" : epoch_disc_loss, "error_g" : epoch_gen_loss, "mask" : mask, "pred": y_fake_thresh_max};

    def __eval_step(self,engine, batch):
        radiograph, mask = batch;
        radiograph,mask = radiograph.to(Config.DEVICE), mask.to(Config.DEVICE)

        #radiograph = (radiograph + 1)/2.0;

        with torch.no_grad():
            y_fake = self.gen(radiograph);

            if Config.NUM_CLASSES > 2:
                y_fake_thresh_max = torch.argmax(y_fake, dim=1).unsqueeze(dim=1);
            else:
                y_fake_thresh_max = y_fake > 0.5;
                
            d_real_list = [];
            d_fake_list = [];

            for i in range(1, Config.NUM_CLASSES):
                y_fake_thresh_max_tmp = y_fake_thresh_max.clone();
                mask_tmp = mask.clone();

                y_fake_thresh_max_tmp[y_fake_thresh_max_tmp != i] = 0;
                y_fake_thresh_max_tmp[y_fake_thresh_max_tmp == i] = 1;
                mask_tmp[mask_tmp != i] = 0;
                mask_tmp[mask_tmp == i] = 1;

                d_fake = self.disc_list[i-1][0](radiograph, y_fake_thresh_max_tmp);
                d_real = self.disc_list[i-1][0](radiograph,mask_tmp);

                d_fake_list.append(d_fake);
                d_real_list.append(d_real);

            loss_gen = self.__loss_func_gen(y_fake, y_fake_thresh_max, mask, d_real_list, d_fake_list, self.ce);
            
        return {"mask" : mask, "loss" : loss_gen, "pred": y_fake_thresh_max};

    def start_train_slot(self, layers_names):
        #ptvsd.debug_this_thread();
        logging.info("Start training...");
        def score_func(engine):
            #loss = engine.state.metrics['loss'];
            metrics = engine.state.metrics;
            columns = list(metrics.keys())
            values = list(metrics.values());

            cm = values[1].cpu().detach().numpy();
            calculate_metrics(cm, columns, values);
            return values[5];
        
        def output_transform(output):
            mask,pred = output['mask'], output['pred'];
            
            if Config.NUM_CLASSES > 2:
                mask = mask.permute(0,2,3,1).long();
                pred = pred.permute(0,2,3,1).long();
                mask = torch.nn.functional.one_hot(mask, num_classes = Config.NUM_CLASSES).squeeze(dim = 3).permute(0,3,1,2);
                pred = torch.nn.functional.one_hot(pred, num_classes = Config.NUM_CLASSES).squeeze(dim = 3).permute(0,3,1,2);
                mask = torch.flatten(mask,2);
                pred = torch.flatten(pred,2);   
            else:
                #pred = pred.round().long();
                pred = igut.to_onehot(pred.long(), 2).squeeze(dim=2);
                pred = torch.flatten(pred, 2);
                #n = pred.detach().numpy();
                mask = mask.long();
                mask = torch.flatten(mask, 1);
            return pred,mask;
        #Remove previously trained model checkpoint
        lst = glob(os.path.sep.join([Config.PROJECT_ROOT,'ckpts',]) + '/*');
        for f in lst:
            os.unlink(f);

        self.initialize_new_train(layers_names);

        self.augmentation_finished_signal.emit();

        #torch.autograd.set_detect_anomaly(True)

        self.trainer = Engine(self.__train_step);
        self.evaluator = Engine(self.__eval_step);
        
        monitoring_metrics_train = ["error_d", "error_g"];
        monitoring_metrics_eval = ["loss"];
        RunningAverage(output_transform=lambda x: x["error_d"]).attach(self.trainer, "error_d")
        RunningAverage(output_transform=lambda x: x["error_g"]).attach(self.trainer, "error_g")

        RunningAverage(output_transform=lambda x: x["loss"]).attach(self.evaluator, "loss")

        if Config.NUM_CLASSES > 2:
            MultiLabelConfusionMatrix(num_classes = Config.NUM_CLASSES, output_transform=output_transform).attach(self.evaluator,'cm');
            MultiLabelConfusionMatrix(num_classes = Config.NUM_CLASSES, output_transform=output_transform).attach(self.trainer,'cm');
        else:
            ConfusionMatrix(num_classes=2, output_transform=output_transform).attach(self.trainer, 'cm');
            ConfusionMatrix(num_classes=2, output_transform=output_transform).attach(self.evaluator, 'cm');

        pbar_trainer = ProgressBar()
        pbar_trainer.attach(self.trainer, metric_names= monitoring_metrics_train);

        pbar_eval = ProgressBar()
        pbar_eval.attach(self.evaluator, metric_names= monitoring_metrics_eval);

        handler = EarlyStopping(patience=25 ,score_function=score_func, trainer = self.trainer);
        model_checkpoint = ModelCheckpoint(os.path.sep.join([Config.PROJECT_ROOT,"ckpts"]),
        n_saved=1,
        filename_prefix=Config.PROJECT_NAME + "_network",
        score_function = score_func,
        global_step_transform=global_step_from_engine(self.trainer),
        require_empty=False);

        self.evaluator.add_event_handler(Events.COMPLETED, handler);
        to_save = {"generator" : self.gen};
        for i in range(len(self.disc_list)):
            to_save[f'disc_{i}'] = self.disc_list[i][0];
        self.evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, to_save = to_save);

        @self.trainer.on(Events.EPOCH_STARTED)
        def epoch_started():
            self.gen.train();
            for d in self.disc_list:
                d[0].train();

        @self.evaluator.on(Events.EPOCH_STARTED)
        def epoch_started():
            self.gen.eval();
            for d in self.disc_list:
                d[0].eval();
    
        @self.trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            
            fname = os.path.join("out", "logs.tsv")
            columns = list(engine.state.metrics.keys());
            values = list(engine.state.metrics.values());
            cm = values[2].cpu().detach().numpy();
            calculate_metrics(cm, columns, values);

            message = f"[{engine.state.epoch}/{Config.NUM_EPOCHS}]"
            for name, value in zip(columns, values):
                if(name != 'cm'):
                    self.writer.add_scalar('training/' + name,float(value),engine.state.epoch);
                    message += f" | {name}: {value}"
            

            pbar_trainer.log_message(message)
            self.update_train_info_epoch_train.emit(columns,values, engine.state.times['EPOCH_COMPLETED'], engine.state.epoch);
            pass

        @self.trainer.on(Events.EPOCH_COMPLETED)
        def log_validattion_results(engine):
            self.evaluator.run(self.valid_loader);
            metrics = self.evaluator.state.metrics;
            columns = list(metrics.keys())
            values = list(metrics.values());

            cm = values[1].cpu().detach().numpy();
            calculate_metrics(cm, columns, values);


            message = f"[{engine.state.epoch}/{Config.NUM_EPOCHS}]"
            for name, value in zip(columns, values):
                if(name != 'cm'):
                    self.writer.add_scalar('validation/' + name,float(value),engine.state.epoch);
                    message += f" | {name}: {value}"

            pbar_trainer.log_message(message)
            self.update_train_info_epoch_valid.emit(columns,values);

            #Save several examples
            save_some_examples(self.gen, self.valid_loader, engine.state.epoch, "evaluation");

        @self.trainer.on(Events.COMPLETED)
        def train_completed():
            self.train_finsihed_signal.emit();
        
        @self.trainer.on(Events.ITERATION_COMPLETED)
        def log_iteration(engine):
            self.update_train_info_iter.emit((engine.state.iteration % engine.state.epoch_length) / engine.state.epoch_length,
             );
            if Config.FINISH_TRAINING is True:
                self.trainer.terminate();
                self.evaluator.terminate();
                Config.FINISH_TRAINING = False;

            
        @self.evaluator.on(Events.ITERATION_COMPLETED)
        def log_iteration(engine):
            self.update_valid_info_iter.emit((engine.state.iteration % engine.state.epoch_length) / engine.state.epoch_length,
             );
        
        #self.trainer.terminate();
        
        self.trainer.run(self.train_loader, Config.NUM_EPOCHS);
    
    def terminate_slot(self):
        self.trainer.terminate();
        self.evaluator.terminate();

    def load_model(self):
        lstdir = glob(os.path.sep.join([Config.PROJECT_ROOT,'ckpts']) + '/*');
        #Find checkpoint file based on extension
        found = False;
        for c in lstdir:
            _, ext = os.path.splitext(c);
            if ext == '.pt':
                found = True;
                #Load train meta to read layer names and number of classes
                self.train_meta = pickle.load(open(os.path.sep.join([Config.PROJECT_ROOT,'ckpts', 'train.meta']),'rb'));
                Config.NUM_CLASSES = self.train_meta[0];
                self.gen.set_num_classes(Config.NUM_CLASSES);
                to_load = {"generator" : self.gen};
                checkpoint_path = c;
                checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE);
                Checkpoint.load_objects(to_load = to_load, checkpoint = checkpoint);

        self.model_loaded_finished.emit(found);
        self.model_load_status = found;

    '''
        Predict of unlabeled data and update the second entry in  dictionary to 1.
    '''
    def predict(self, lbl, dc):
        #Because predicting is totally based on the initialization of the model,
        # if we haven't loaded a model yet or the loading wasn't successfull
        # we should not do anything and return immediately.
        if self.model_load_status:
            self.gen.eval();
            with torch.no_grad():
                radiograph_image = cv2.imread(os.path.sep.join([Config.PROJECT_ROOT, 'images', lbl]),cv2.IMREAD_GRAYSCALE);
                clahe = cv2.createCLAHE(5,(9,9));
                radiograph_image = clahe.apply(radiograph_image);
                
                w,h = radiograph_image.shape;
                transformed = self.valid_transforms(image = radiograph_image);
                radiograph_image = transformed["image"];
                radiograph_image = torch.unsqueeze(radiograph_image,0);
                radiograph_image = radiograph_image.to(Config.DEVICE);
                p = self.gen(radiograph_image);
                mask_list = [];
                if Config.NUM_CLASSES > 2:
                    num_classes = p.size()[1];
                    p = p.permute(0,2,3,1).cpu().detach().numpy()[0];
                    p = np.argmax(p, axis=2);
                    
                    #Convert each class to a predefined color
                    for i in range(1,num_classes):
                        mask = np.zeros(shape=(Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3),dtype=np.uint8);
                        tmp = (p==i);
                        mask[(tmp)] = PIL.ImageColor.getrgb(Config.PREDEFINED_COLORS[i-1]);
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
