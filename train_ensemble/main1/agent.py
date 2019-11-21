#NATIVE LIBRARY IMPORTS
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

#######################################################################################################
#######################################################################################################

#REPLAY MEMORY FOR TRAINING OR MODEL SELECTION NETWORK
Transition = namedtuple('Transition',('state', 'action', 'reward', 'done'))
class ReplayMemory(object):
    def __init__(self, capacity,device='cpu'):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.device=device

    def push(self,state,action,reward,done):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        state = state.cpu().squeeze(0)
        action = torch.Tensor([action]).long()
        reward = torch.Tensor([reward])
        done = torch.Tensor([done])

        self.memory[self.position] = Transition(state,action,reward,done)
        self.position = (self.position + 1) % self.capacity

    def sample(self,batch_size):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=batch_size)
        s = [None] * batch_size
        a = [None] * batch_size
        r = [None] * batch_size
        for i, e in enumerate(experiences):
            s[i] = e.state
            a[i] = e.action
            r[i] = e.reward

        s = torch.stack(s).to(self.device)
        a = torch.stack(a).to(self.device)
        r = torch.stack(r).to(self.device)
        return s,a,r

    def __len__(self):
        return len(self.memory)

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
    def __init__(self,k,upsize):
        super(Model,self).__init__()

        #ZERO OUT THE WEIGHTS
        def zeroout(m):
            with torch.no_grad():
                if type(m) == torch.nn.Conv2d:
                    m.weight.data *= 0.0
        self.SegNet = models.segmentation.deeplabv3_resnet101(pretrained=False,num_classes=k)

        #self.first = torch.nn.Sequential(
        #        torch.nn.BatchNorm2d(3),
        #        torch.nn.Conv2d(3,32,3,1,1)
        #        )

        #self.db1 = DenseBlock(32,8)
        #self.db2 = DenseBlock(32*2,8)
        #self.db3 = DenseBlock(32*3,8)
        #self.db4 = DenseBlock(32*4,8)
        #[b.apply(init_weights) for b in [self.db1,self.db2,self.db3,self.db4]]

        #if upsize == 4:
        #    self.final = torch.nn.Sequential(
        #            torch.nn.ConvTranspose2d(64,64,4,2,1,bias=False),
        #            torch.nn.BatchNorm2d(64),
        #            torch.nn.PReLU(),
        #            torch.nn.ConvTranspose2d(64,32,4,2,1,bias=False),
        #            torch.nn.BatchNorm2d(32),
        #            torch.nn.PReLU(),
        #            torch.nn.Conv2d(32,k,3,1,1)
        #            )
        if upsize == 4:
            self.final = torch.nn.Upsample(scale_factor=upsize)
        elif upsize == 8:
            self.final = torch.nn.Sequential(
                    torch.nn.Upsample(scale_factor=2),
                    torch.nn.Conv2d(160,64,3,1,1),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.PReLU(),
                    torch.nn.Upsample(scale_factor=2),
                    torch.nn.Conv2d(64,64,3,1,1),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.PReLU(),
                    torch.nn.Upsample(scale_factor=2),
                    torch.nn.Conv2d(64,32,3,1,1),
                    torch.nn.BatchNorm2d(32),
                    torch.nn.PReLU(),
                    torch.nn.Conv2d(32,k,3,1,1)
                    )
        self.softmaxfn = torch.nn.Softmax(dim=1)

    #FORWARD FUNCTION
    def forward(self,x):
        x = self.SegNet(x)['out']
        #x = self.first(x)
        #x = self.db1(x)
        #x = self.db2(x)
        #x = self.db3(x)
        #x = self.db4(x)
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
        self.model = Model(self.ACTION_SPACE,args.upsize)
        self.model.to(self.device)
        if chkpoint:
            self.model.load_state_dict(chkpoint['agent'])
        else:
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

