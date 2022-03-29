#===============================================================
#===============================================================
import torch
import torch.nn as nn
import torch.functional as F
from torch.nn.functional import one_hot, binary_cross_entropy_with_logits
#===============================================================
#===============================================================

#===============================================================
def dice_loss(logits, 
                true, 
                eps=1e-7, 
                sigmoid = False,
                multilabel = False,
                arange_logits = False):

    if sigmoid is True:
        logits = torch.sigmoid(logits);
    
    if arange_logits is True:
        logits = logits.permute(0,2,3,1);

    dims = (1,2,3);

    intersection = torch.sum(true * logits, dims);
    union = torch.sum(true + logits, dims);
    d_loss = torch.mean((2.0*intersection) / (union + eps));
    return 1-d_loss;
#===============================================================

#===============================================================
def focal_loss(logits,
                true,
                alpha = 0.8,
                gamma = 2.0,
                arange_logits = False):

    if arange_logits is True:
        logits = logits.permute(0,2,3,1);
    
    bce = binary_cross_entropy_with_logits(logits, true.float(), reduction='none');
    bce_exp = torch.exp(-bce);
    f_loss = torch.mean(alpha * (1-bce_exp)**gamma*bce);
    return f_loss;
#===============================================================

#===============================================================
def tversky_loss(logits,
                true,
                alpha = 0.5,
                beta = 1.0,
                sigmoid = False,
                arange_logits = False,
                smooth = 1):

    if arange_logits is True:
        logits = logits.permute(0,2,3,1);
    
    if sigmoid is True:
        logits = torch.sigmoid(logits);
    
    dims = (1,2,3);
    tp = torch.sum(true * logits, dims);
    fp = torch.sum((1-true) * logits, dims);
    fn = torch.sum(true * (1-logits), dims);
    tversky = torch.mean((tp + smooth) / (tp + alpha*fp + beta*fn + smooth))  
    return 1-tversky;
#===============================================================