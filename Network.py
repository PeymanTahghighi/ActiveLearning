import torch
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU, ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import ConvTranspose2d
from dropblock import DropBlock2D
from torch.nn.functional import log_softmax, conv_transpose2d
import Config

class CNNBlock(nn.Module):
    def __init__(self, inChannels, outChannels, stride):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel_size=4, padding=1, stride=stride, bias=False, padding_mode="reflect",),
            nn.BatchNorm2d(outChannels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x);


class Discriminator(nn.Module):
    def __init__(self, inChannels=1, features = [64,128,256,512]):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(inChannels, features[0], kernel_size=4, stride=2, padding = 1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )

        self.l2 = CNNBlock(features[0], features[1], stride=2);
        self.l3 = CNNBlock(features[1], features[2], stride=2);

        # layers = [];
        # inChannels = features[0];
        # for numFeatures in features[1:]:
        #     layers.append(CNNBlock(inChannels=inChannels,
        #     outChannels=numFeatures, stride = 1 if numFeatures == features[-1] else 2));

        #     inChannels = numFeatures;
        
        # layers.append(nn.Conv2d(inChannels, 1, kernel_size=4, padding=1, stride=1, padding_mode=
        # "reflect"));
        # self.model = nn.Sequential(*layers);

    
    def forward(self, radiograph, mask):
        seg = torch.mul(mask, radiograph);
        #cat = seg;

        out_l1 = self.l1(seg);
        out_l2 = self.l2(out_l1);
        out_l3 = self.l3(out_l2);
        
        cat0 = seg.view(radiograph.size()[0],-1);
        cat1 = out_l1.view(radiograph.size()[0],-1);
        cat2 = out_l2.view(radiograph.size()[0],-1);
        cat3 = out_l3.view(radiograph.size()[0],-1);
        cat = torch.cat((cat0, cat1, cat2, cat3), dim=1);
        return cat;

class Block(nn.Module):
    def __init__(self, inChannels, outChannels, down=True, act = "relu", dropout = False):
        super().__init__();

        if down:
            self.conv = nn.Sequential(
                nn.Conv2d(inChannels, outChannels, stride=2, padding=1, kernel_size=4, bias=False, padding_mode="reflect"),
                DropBlock2D(drop_prob=0.2, block_size=4),
                #nn.Dropout2d(0.5),
                nn.BatchNorm2d(outChannels),
                nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2)
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(inChannels, outChannels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(outChannels),
                nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2)
            )

        self.droptout = dropout;
        self.droptout = nn.Dropout(0.5);
    def forward(self, x):
        x = self.conv(x);
        return x;

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__();
        self.num_classes = 1;
        self.out_channels = out_channels;
        self.in_channels = in_channels;

        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        self.down1 = Block(out_channels, out_channels*2, down=True, act="leaky", dropout=False);
        self.down2 = Block(out_channels*2, out_channels*4, down=True, act="leaky", dropout=False);
        self.down3 = Block(out_channels*4, out_channels*8, down=True, act="leaky", dropout=False);
        self.down4 = Block(out_channels*8, out_channels*8, down=True, act="leaky", dropout=False);
        self.down5 = Block(out_channels*8, out_channels*8, down=True, act="leaky", dropout=False);
        self.down6 = Block(out_channels*8, out_channels*8, down=True, act="leaky", dropout=False);

        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_channels*8, out_channels*8, 4, 2, 1, padding_mode="reflect"),
            nn.ReLU()
        )

        self.up1 = Block(out_channels*8, out_channels*8, down=False, act="relu", dropout= True);
        self.up2 = Block(out_channels*8*2, out_channels*8, down=False, act="relu", dropout= True);
        self.up3 = Block(out_channels*8*2, out_channels*8, down=False, act="relu", dropout= True);
        self.up4 = Block(out_channels*8*2, out_channels*8, down=False, act="relu", dropout= False);
        self.up5 = Block(out_channels*8*2, out_channels*4, down=False, act="relu", dropout= False);
        self.up6 = Block(out_channels*4*2, out_channels*2, down=False, act="relu", dropout= False);
        self.up7 = Block(out_channels*2*2, out_channels, down=False, act="relu", dropout= False);
        
    def set_num_classes(self, c):
        self.num_classes = c;
        #Set last layers according to the number of classes
        self.final = nn.Sequential(
            nn.ConvTranspose2d(self.out_channels*2, 1 if c==2 else c, kernel_size=4, padding=1, stride=2),
            nn.LogSoftmax(dim=1) if c > 2 else nn.Sigmoid()
        ).to(Config.DEVICE);
    
    def assert_nan(self,tensor):
        assert(torch.isnan(tensor).all() == False);

    def forward(self, x):
        d1 = self.initial_down(x);
        #self.assert_nan(d1);
        
        d2 = self.down1(d1);
       # self.assert_nan(d2);
        d3 = self.down2(d2);
        #self.assert_nan(d3);
        d4 = self.down3(d3);
       # self.assert_nan(d4);
        d5 = self.down4(d4);
        #self.assert_nan(d5);
        d6 = self.down5(d5);
        #self.assert_nan(d6);
        d7 = self.down6(d6);
        #self.assert_nan(d7);
        d8 = self.bottleneck(d7);
      #  self.assert_nan(d8);

        u1 = self.up1(d8);
       # self.assert_nan(u1);
        u2 = self.up2(torch.cat([u1, d7], dim=1));
        #self.assert_nan(u2);
        u3 = self.up3(torch.cat([u2, d6], dim=1));
        #self.assert_nan(u3);
        u4 = self.up4(torch.cat([u3, d5], dim=1));
       # self.assert_nan(u4);
        u5 = self.up5(torch.cat([u4, d4], dim=1));
       # self.assert_nan(u5);
        u6 = self.up6(torch.cat([u5, d3], dim=1));
       # self.assert_nan(u6);
        u7 = self.up7(torch.cat([u6, d2], dim=1));
      #  self.assert_nan(u7);

        final = self.final(torch.cat([u7, d1], dim=1));
       # self.assert_nan(final);
        return final;

def test():
    x = torch.randn((1,3,512,512));
    model = Generator(inChannels=3, outChannels=64);
    ret = model(x);
    print(ret.shape);
        
if __name__ == '__main__':
    test();