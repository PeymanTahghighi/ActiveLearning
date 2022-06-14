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
    x, y = next(iter(val_loader))
    x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
    with torch.no_grad():
        output, _ = model(x)
        output = (torch.sigmoid(output) > 0.5).permute(0,2,3,1);
        b_size = output.size()[0];

        for b in range(b_size):
            b_output = output[b];

            output_colored = torch.zeros((Config.NUM_CLASSES, b_output.shape[0], b_output.shape[1], 3)).long().to(Config.DEVICE);
            for i in range(Config.NUM_CLASSES):
                col = np.full((Config.IMAGE_SIZE,Config.IMAGE_SIZE, 3), [Config.PREDEFINED_COLORS[i][0], 
                Config.PREDEFINED_COLORS[i][1], 
                Config.PREDEFINED_COLORS[i][2]]);

                output_cls = b_output[:,:,i];
                
                col = torch.tensor(col).to(Config.DEVICE).long();
                a = output_colored[i];
                cond = (output_cls == 1).bool().unsqueeze(axis = 2);
                output_colored[i] = torch.where(cond, col, output_colored[i]);

            output_grid = make_grid(output_colored.permute(0,3,1,2), Config.NUM_CLASSES);
            save_image(output_grid.float(), os.path.sep.join([Config.PROJECT_ROOT, folder, f"input_{epoch}-{b}.png"]))

        if epoch == 1:
            radiograph_grid = make_grid(x*0.229 + 0.485, b_size)
            save_image(radiograph_grid, os.path.sep.join([Config.PROJECT_ROOT, folder, f"radiograph.png"]))
            #gt_grid = make_grid(y.float(), Config.BATCH_SIZE)
            #save_image(gt_grid, os.path.sep.join([Config.PROJECT_ROOT, folder, f"gt.png"]))

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