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
#import torchvision.models as models
import deeplabv3_resnet50 as models

# custom imports
import RRDBNet_arch as RRDB
import unet_model

#######################################################################################################
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

        # CREATE OUR DENSEBLOCK WHICH ADDS GROWTH FEATURE CHANNELS TO GLOBAL
        self.block1 = torch.nn.Sequential(
                torch.nn.Conv2d(channel_in,k,5,1,2,bias=False),
                torch.nn.ReLU()
                )
        self.block2 = torch.nn.Sequential(
                torch.nn.Conv2d(channel_in + 1*k,k,5,1,2,bias=False),
                torch.nn.ReLU()
                )
        self.block3 = torch.nn.Sequential(
                torch.nn.Conv2d(channel_in + 2*k,k,5,1,2,bias=False),
                torch.nn.ReLU()
                )
        self.block4 = torch.nn.Sequential(
                torch.nn.Conv2d(channel_in + 3*k,channel_in,5,1,2,bias=False),
                torch.nn.ReLU()
                )

    # forward function
    def forward(self,x):
        f = torch.cat((x,self.block1(x)),1)
        f = torch.cat((f,self.block2(f)),1)
        f = torch.cat((f,self.block3(f)),1)
        x = x + self.block4(f)
        return x

#OUR ENCODER DECODER NETWORK FOR MODEL SELECTION NETWORK
class Model(nn.Module):
    def __init__(self,k):
        super(Model,self).__init__()

        #self.SegNet = models.segmentation.fcn_resnet101(pretrained=True,num_classes=21)
        self.SegNet = models.deeplabv3_resnet50(pretrained=False,num_classes=k)


        self.first = torch.nn.Sequential(
                torch.nn.Conv2d(k*3,64,5,1,2,bias=False)
                )

        self.db1 = DenseBlock(64,24)
        self.db2 = DenseBlock(64,24)
        self.db3 = DenseBlock(64,24)

        self.final = torch.nn.Sequential(
                torch.nn.Conv2d(64,64,5,1,2,bias=False),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64,k,5,1,2,bias=False),
                torch.nn.ReLU(),
                torch.nn.Softmax(dim=1)
                )

    #FORWARD FUNCTION
    def forward(self,x):
        x = self.SegNet(x)['out']
        x = F.softmax(x,dim=1)
        #x = self.first(x)
        #x = self.db1(x)
        #x = self.db2(x)
        #x = self.db3(x)
        #x = self.final(x)
        return x

#######################################################################################################

#AGENT COMPRISES OF A MODEL SELECTION NETWORK AND MAKES ACTIONS BASED ON IMG PATCHES
class Agent():
    def __init__(self,args,train=True,chkpoint=None):

        def init_weights(m):
            if isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.fill_(0.01)

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
        #self.model = Model(self.ACTION_SPACE)
        self.model = unet_model.UNet(self.ACTION_SPACE,bilinear=False)
        self.model.to(self.device)
        if chkpoint:
            print('Agent Loaded at checkpoint!')
            self.model.load_state_dict(chkpoint['agent'])

        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(),lr=1e-4)
        #self.opt = torch.optim.SGD(self.model.parameters(),lr=1e-3)
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



