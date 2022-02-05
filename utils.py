import torch
import torch.nn as nn
from torch.types import Device
import Config
from torchvision.utils import make_grid, save_image
import os
import cv2
import numpy as np

def gradient_penalty(critic, real, fake, image):
    _,c,h,w = real.shape;
    epsilon = torch.randint(0, 1, (Config.BATCH_SIZE,1,1,1)).repeat(1,c,h,w).to(Config.DEVICE);
    #epsilon.require_grad = True;
    epsilon = torch.tensor(epsilon, dtype=torch.float, requires_grad=True);
    interpolated_image = real*epsilon + fake * (1-epsilon);
    # interpolated_image_np = np.array(real.permute(0,2,3,1).cpu().detach().numpy(),np.uint8);
    # print(interpolated_image_np);
    # cv2.imshow("t",interpolated_image_np[0]*255);
    # cv2.waitKey();


    mixed_scores = critic(image, interpolated_image);
    gradient = torch.autograd.grad(
        inputs=interpolated_image,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0];

    gradient = gradient.view(gradient.shape[0],-1);
    gradient_norm = gradient.norm(2,dim=1);
    gradient_penalty = torch.mean((gradient_norm - 1)**2);
    return gradient_penalty * 10.0;

def cross_entropy(p,q):
    return np.sum(-p*np.log(q));

def JSD(p,q):
    p = p + Config.EPSILON;
    q = q + Config.EPSILON;
    avg = (p+q)/2;
    jsd = (cross_entropy(p,avg) - cross_entropy(p,p))/2 + (cross_entropy(q,avg) - cross_entropy(q,q))/2;
    #clamp
    if jsd > 1.0:
        jsd = 1.0;
    elif jsd < 0.0:
        jsd = 0.0;
    
    return jsd;

def save_samples(model, val_loader, epoch, folder):
    x, y, _ = next(iter(val_loader))
    x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
    with torch.no_grad():
        y_fake, _ = model(x)
        if Config.NUM_CLASSES > 2:
            y_fake = torch.argmax(y_fake,dim=1).unsqueeze(dim = 1);
        else:
            y_fake = (y_fake > 0.5).float();
        
        y_fake_colored = torch.zeros((Config.BATCH_SIZE, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)).long().to(Config.DEVICE);
        for i in range(1,Config.NUM_CLASSES):
            col = np.full((Config.BATCH_SIZE,Config.IMAGE_SIZE,Config.IMAGE_SIZE, Config.NUM_CLASSES), [Config.PREDEFINED_COLORS[i-1][0], 
            Config.PREDEFINED_COLORS[i-1][1], 
            Config.PREDEFINED_COLORS[i-1][2]]);
            col = torch.tensor(col).to(Config.DEVICE).permute(0,3,1,2).long();
            cond = (y_fake!=i).bool();
            y_fake_colored = y_fake_colored.where(cond, col);
            # n = y_fake_colored.permute(0,2,3,1,).cpu().detach().numpy();
            # n = np.array(n,dtype=np.uint8);
            # cv2.imshow('t',n[0]);
            # cv2.waitKey();
        # a = y_fake[y_fake == 1];
        # y_fake_colored[a == 1] = 50;
        #n = y_fake_colored.permute(0,2,3,1,).cpu().detach().numpy();
        #cv2.imshow('t',n[0]);
        #cv2.waitKey();
        #y_fake_colored[a] = 255;
        fake_grid = make_grid(y_fake_colored, Config.BATCH_SIZE);
        save_image(fake_grid.float(), os.path.sep.join([Config.PROJECT_ROOT, folder, f"input_{epoch}.png"]))

        if epoch == 1:
            radiograph_grid = make_grid(x *0.5+ 0.5, Config.BATCH_SIZE)
            save_image(radiograph_grid, os.path.sep.join([Config.PROJECT_ROOT, folder, f"radiograph.png"]))
            gt_grid = make_grid(y.float(), Config.BATCH_SIZE)
            save_image(gt_grid, os.path.sep.join([Config.PROJECT_ROOT, folder, f"gt.png"]))

def save_checkpoint(model, epoch):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "epoch" : epoch
    }
    torch.save(checkpoint, os.path.sep.join([Config.PROJECT_ROOT, "ckpts", f"ckpt.pt"]))

def load_checkpoint(checkpoint_file, model):
    if(os.path.exists(checkpoint_file)):

        print("=> Loading checkpoint")
        
        checkpoint = torch.load(checkpoint_file, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint["state_dict"])

    return 0;

def pixmap_to_numpy(pixmap):
    image = pixmap.toImage();
    img_arr = np.fromstring(image.bits().asstring(image.width() * image.height() * 4), 
    dtype=np.uint8).reshape((image.height(), image.width(), 4))
    img_arr = img_arr[:,:,:3]; 
    return img_arr;