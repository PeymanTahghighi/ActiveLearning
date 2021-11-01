import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, plot_precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
from torch.functional import Tensor
import Config

def calculate_metrics(cm : np.array, columns, values):
    #If we are doing multi label segmentation
    if Config.NUM_CLASSES > 2:
        #For each of the classes calculate the metrics and then average them
        running_accuracy = 0;
        running_sensitivity = 0;
        runing_precision = 0;
        running_f1 = 0;
        for i in range(cm.shape[0]):
            cm_tmp = cm[i];
            total = np.sum(cm_tmp);
            running_accuracy += (cm_tmp[0,0] + cm_tmp[1,1]) / total;
            tp = cm_tmp[1,1];
            fn = cm_tmp[1,0];
            fp = cm_tmp[0,1];
            tn = cm_tmp[0,0];
            pr = (tp) /(tp + fp + Config.EPSILON);
            sens = (tp) /(tp + fn + Config.EPSILON);
            f1 = 2*pr*sens / (pr + sens + Config.EPSILON);
            running_sensitivity += sens;
            runing_precision += pr;
            running_f1 += f1;

        accuracy = running_accuracy / cm.shape[0];
        precision = runing_precision/ cm.shape[0];
        sensitivity = running_sensitivity / cm.shape[0];
        F1 = running_f1 / cm.shape[0];
        columns.append('accuracy');
        columns.append('sensitivity');
        columns.append('precision');
        columns.append('f1');
        values.append(accuracy);
        values.append(sensitivity);
        values.append(precision);
        values.append(F1);
    else:
        tp = cm[1,1];
        fn = cm[1,0];
        fp = cm[0,1];
        tn = cm[0,0];
        total = np.sum(cm);
        pr = tp / (tp + fp + Config.EPSILON);
        rec = tp / (tp + fn + Config.EPSILON);
        f1 = 2*pr*rec / (pr + rec + Config.EPSILON);
        accuracy = (tp + tn) / total;

        columns.append('accuracy');
        columns.append('sensitivity');
        columns.append('precision');
        columns.append('f1');
        values.append(accuracy);
        values.append(rec);
        values.append(pr);
        values.append(f1);



# def conf_matrix(y,pred):
#     #pred = np.argmax(pred, 3);
    
#     print(cm);


# def true_positives(y,pred,th = 0.5):
#     thresh = pred >= th;
#     tp = torch.sum((thresh == 1) & (y == 1));
#     return tp;

# def true_negatives(y,pred,th = 0.5):
#     thresh = pred >= th;
#     tn = torch.sum((thresh == 0) & (y == 0));
#     return tn;

# def false_negatives(y,pred,th = 0.5):
#     thresh = pred >= th;
#     fn = torch.sum((thresh == 0) & (y == 1));
#     return fn;

# def false_positives(y,pred,th = 0.5):
#     thresh = pred >= th;
#     fp = torch.sum((thresh == 1) & (y == 0));
#     return fp;

# def sensitivity(y,pred,th = 0.5):
#     tp = true_positives(y,pred,th);
#     fn = false_negatives(y,pred,th);
#     return tp/(tp+fn);

# def specificity(y,pred,th=0.5):
#     tn = true_negatives(y,pred,th);
#     fp = false_positives(y,pred,th);
#     return tn/(tn+fp);

# def accuracy(y,pred,th = 0.5):
#     tp = true_positives(y,pred,th);
#     fp = false_positives(y,pred,th);
#     fn = false_negatives(y,pred,th);
#     tn = true_negatives(y,pred,th);
#     return (tp+tn) / (fp+tn+tp+fn);

# def PPV(y,pred,th=0.5):
#     tp = true_positives(y,pred,th);
#     fp = false_positives(y,pred,th);
#     return tp/(tp+fp);

# def get_roc_curve(y,pred, roc_func,epoch, th = 0.5):
    
#     auc_roc = roc_auc_score(y,pred);
#     label = "AUC: %0.3f" %auc_roc;
#     a,b,_ = roc_func(y,pred);
#     plt.figure(epoch,figsize=(7,7));
#     plt.plot([0,0],[1,1],'k--');
#     plt.plot(a,b,label = label);
#     plt.legend(loc='best');
#     plt.savefig('ROC'+ str(epoch) +'.png');

# def get_prc_curve(y, pred, prc_func,epoch, th = 0.5):
#     auc_prc = average_precision_score(y,pred);
#     label = "AUCPRC:%0.4f" % auc_prc;
#     a,b,_ = prc_func(y,pred);
    
#     plt.figure(epoch + NUM_EPOCHS, figsize=(7, 7))
#     plt.step(b, a, where='post', label=label)
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.ylim([0.0, 1.05])
#     plt.xlim([0.0, 1.0])
#     plt.legend(loc='best')

#     plt.savefig("PRC" + str(epoch) + ".png");
