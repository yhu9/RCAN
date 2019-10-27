#NATIVE LIBRARY IMPORTS
import functools
import os
import random
import math
from collections import namedtuple
from collections import OrderedDict

#OPEN SOURCE IMPORTS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# custom imports
import RRDBNet_arch as RRDB

#######################################################################################################
#######################################################################################################

class SRSmall(nn.Module):
    def __init__(self, channel=3, upsize=4, k=8):

        self.head = torch.nn.Sequential(
                torch.nn.Conv2d(channel,32,3,1,1),
                torch.nn.ReLU()
                )

        self.block1 = torch.nn.Sequential(
                torch.nn.Conv2d(32,k,3,1,1),
                torch.nn.ReLU()
                )
        self.block2 = torch.nn.Sequential(
                torch.nn.Conv2d(32,k,3,1,1),
                torch.nn.ReLU()
                )
        self.block3 = torch.nn.Sequential(
                torch.nn.Conv2d(32,k,3,1,1),
                torch.nn.ReLU(),
                )
        self.block4 = torch.nn.Conv2d(32+3*k,32,1,1)

        self.tail = torch.nn.Sequential(
                torch.nn.Conv2d(channel,32,3,1,1),
                torch.nn.ReLU()
                )

    def forward(self,x):
        x = self.head(x)
        x = torch.cat((x,self.block1(x)),1)
        x = torch.cat((x,self.block2(x)),1)
        x = torch.cat((x,self.block3(x)),1)
        x = self.block4(x) + x[:,:32]

#######################################################################################################
#######################################################################################################
#DENSE NET ARCHITECTURE WITHOUT BATCH NORMALIZATION
class DenseBlock(nn.Module):
    def __init__(self,channel_in,k):
        super(DenseBlock,self).__init__()

        #RANDOM MODEL INITIALIZATION FUNCTION
        def init_weights(m):
            if isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)

        #CREATE OUR DENSEBLOCK WITCH ADDS GROWTH FEATURE CHANNELS TO GLOBAL
        self.block1 = torch.nn.Sequential(
                torch.nn.BatchNorm2d(channel_in),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(channel_in,4*k,3,1,0),
                torch.nn.BatchNorm2d(4*k),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(4*k,k,3,1,0)
                )
        self.block2 = torch.nn.Sequential(
                torch.nn.BatchNorm2d(channel_in + 1*k),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(channel_in + 1*k,4*k,3,1,0),
                torch.nn.BatchNorm2d(4*k),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(4*k,k,3,1,0)
                )
        self.block3 = torch.nn.Sequential(
                torch.nn.BatchNorm2d(channel_in + 2*k),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(channel_in + 2*k,4*k,3,1,0),
                torch.nn.BatchNorm2d(4*k),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(4*k,k,3,1,0)
                )
        self.block4 = torch.nn.Sequential(
                torch.nn.BatchNorm2d(channel_in + 3*k),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(channel_in + 3*k,4*k,3,1,0),
                torch.nn.BatchNorm2d(4*k),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(4*k,k,3,1,0)
                )
        self.block1.apply(init_weights)
        self.block2.apply(init_weights)
        self.block3.apply(init_weights)
        self.block4.apply(init_weights)

    #FORWARD FUNCTION
    def forward(self,x):
        features = self.block1(x)
        x = torch.cat((features,x),1)
        features = self.block2(x)
        x = torch.cat((features,x),1)
        features = self.block3(x)
        x = torch.cat((features,x),1)
        features = self.block4(x)
        x = torch.cat((features,x),1)
        return x

#OUR ENCODER DECODER NETWORK FOR MODEL SELECTION NETWORK
class Model(nn.Module):
    def __init__(self,k):
        super(Model,self).__init__()

        #ZERO OUT THE WEIGHTS
        def zeroout(m):
            with torch.no_grad():
                if type(m) == torch.nn.Conv2d:
                    m.weight.data *= 0.0

        #self.SegNet = models.segmentation.fcn_resnet50(pretrained=False,num_classes=48)
        #self.SegNet = models.segmentation.deeplabv3_resnet101(pretrained=False,num_classes=160)

        self.first = torch.nn.Sequential(
                torch.nn.BatchNorm2d(3),
                torch.nn.Conv2d(3,32,3,1,1)
                )

        self.db1 = DenseBlock(32,8)
        self.db2 = DenseBlock(32*2,8)
        self.db3 = DenseBlock(32*3,8)
        self.db4 = DenseBlock(32*4,8)
        #[b.apply(init_weights) for b in [self.db1,self.db2,self.db3,self.db4]]

        self.final = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2),
                torch.nn.Conv2d(160,64,3,1,1),
                torch.nn.BatchNorm2d(64),
                torch.nn.PReLU(),
                torch.nn.Upsample(scale_factor=2),
                torch.nn.Conv2d(64,32,3,1,1),
                torch.nn.BatchNorm2d(32),
                torch.nn.PReLU(),
                torch.nn.Conv2d(32,k,3,1,1)
                #torch.nn.Softmax(dim=1)
                )
        '''
        self.final = torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(160,64,3,1),
                torch.nn.ConvTranspose2d(64,64,2,2),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(64,32,3,1),
                torch.nn.ConvTranspose2d(32,32,2,2),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(32,k,3,1),
                torch.nn.Softmax(dim=1)
                )
        '''
        self.softmaxfn = torch.nn.Softmax(dim=1)

    #FORWARD FUNCTION
    def forward(self,x):
        #x = self.SegNet(x)['out']
        x = self.first(x)
        x = self.db1(x)
        x = self.db2(x)
        x = self.db3(x)
        x = self.db4(x)
        x = self.final(x)
        x = self.softmaxfn(x)
        return x

    #FORWARD FUNCTION
    def raw(self,x):
        #x = self.SegNet(x)['out']
        x = self.first(x)
        x = self.db1(x)
        x = self.db2(x)
        x = self.db3(x)
        x = self.db4(x)
        x = self.final(x)
        return x

#######################################################################################################

#AGENT COMPRISES OF A MODEL SELECTION NETWORK AND MAKES ACTIONS BASED ON IMG PATCHES
class Agent():
    def __init__(self,args,train=True,chkpoint=None):

        def init_weights(m):
            if isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
        #INITIALIZE HYPER PARAMS
        self.device = args.device
        self.ACTION_SPACE = args.action_space
        if train:
            self.steps = 0
            self.BATCH_SIZE = args.batch_size
            self.GAMMA = args.gamma
            self.EPS_START = args.eps_start
            self.EPS_END = args.eps_end
            self.EPS_DECAY = args.eps_decay
            self.TARGET_UPDATE = args.target_update

        #INITIALIZE THE MODELS
        self.model = Model(self.ACTION_SPACE)
        self.model.to(self.device)
        self.model.apply(init_weights)

        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(),lr=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt,200,0.5)

#######################################################################################################
#######################################################################################################

#SOME TESTING CODE TO MAKE SURE THIS FILE WORKS
if __name__ == "__main__":

    device = 'cuda'
    m = Model()
    m.to(device)
    img = torch.rand((3,100,100)).unsqueeze(0).to(device)

    print('Img shape: ', img.shape)
    print('out shape: ', m(img).shape)
    print('encode shape: ', m.encode(img).shape)
    print('decode shape: ', m.decode(m.encode(img)).shape)

