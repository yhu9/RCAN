#NATIVE LIBRARY IMPORTS
import os
import random
import math
from collections import namedtuple

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

#WEIGHT MATRIX M
class Mask(nn.Module):
    def __init__(self,num_patches,action_space):
        super(Mask,self).__init__()

        #DEFINE PARAMETER MATRIX of size NxK and fill it with equal probability in [0,1]
        self.weight = torch.nn.Parameter(data=torch.Tensor(num_patches, action_space), requires_grad=True)
        self.weight.data.fill_(1/action_space)
        self.fn = torch.nn.Softmax(dim=1)

    def forward(self,x,labels):
        probs = self.fn(self.weight[labels])
        myweights = probs.mul(x)
        return myweights

class M(nn.Module):
    def __init__(self,num_patches,action_space):
        super(M,self).__init__()
        self.M = Mask(num_patches,action_space)

    def forward(self,x,labels):
        weights = self.M(x,labels)
        return weights

#OUR ENCODER DECODER NETWORK FOR MODEL SELECTION NETWORK
class Model(nn.Module):
    def __init__(self,action_space=10):
        super(Model,self).__init__()
        #RANDOM MODEL INITIALIZATION FUNCTION
        def init_weights(m):
            if isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)

        self.encoder = models.resnet18(pretrained=True)
        self.decoder = torch.nn.Sequential(
                    nn.Linear(1000,512),
                    nn.ReLU(),
                    nn.Linear(512,256),
                    nn.ReLU(),
                    nn.Linear(256,action_space)
                )
        self.decoder.apply(init_weights)

    def encode(self,x):
        return torch.tanh(self.encoder(x))

    def decode(self,x):
        return self.decoder(x)

    def forward(self,x):
        latent_vector = self.encode(x)
        out = self.decode(latent_vector)
        return out

#######################################################################################################

#AGENT COMPRISES OF A MODEL SELECTION NETWORK AND MAKES ACTIONS BASED ON IMG PATCHES
class Agent():
    def __init__(self,args,train=True,chkpoint=None):

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
            self.memory = ReplayMemory(args.memory_size,device=self.device)

        #INITIALIZE THE MODELS
        self.model = Model(action_space=self.ACTION_SPACE)
        self.M = {}
        self.O = {}
        if chkpoint:
            self.model.load_state_dict(chkpoint['agent'])

        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(),lr=0.0001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt,200,0.99)

    #FUNCTION TO CREATE NEW WEIGHT MATRIX A
    def createM(self,n,k,SRmodels,idx):
        A = M(n,k)
        A.to(self.device)
        self.M[idx] = A
        optA = torch.optim.Adam(A.parameters(),lr=0.01)
        self.O[idx] = optA

    def trainM(self,SRmodels,SRoptimizers,LR_DATA,HR_DATA):

        self.createM(len(LR_DATA), self.ACTION_SPACE, SRmodels)

        loss = 1; c = 0.01
        patch_ids = list(range(len(LR_DATA)))
        while loss > c:
            batch_ids = random.sample(patch_ids,self.batch_size)    #TRAIN ON A SINGLE IMAGE

            labels = torch.Tensor(batch_ids).long().to(self.device)
            lrbatch = LR_DATA[labels,:,:,:]
            hrbatch = HR_DATA[labels,:,:,:]

            self.O[-1].zero_grad()
            for i,sisr in enumerate(SRmodels):
                SRoptimizers[i].zero_grad()           #zero our sisr gradients
                sr = sisr(lrbatch)

                #update sisr model based on weighted l1 loss
                l1diff = torch.abs(sr - hrbatch).view(len(batch_ids),-1).mean(1)           #64x1 vector
                onehot = torch.zeros(self.SR_COUNT); onehot[j] = 1.0                #1x4 vector
                imgscore = torch.matmul(l1diff.unsqueeze(1),onehot.to(self.device).unsqueeze(0))    #64x4 matrix with column j as l1 diff and rest as zeros

                weighted_imgscore = self.M[-1](imgscore,labels)     #do element wise matrix multiplication of l1 diff and softmax weights
                loss1 = torch.sum(weighted_imgscore)
                loss1.backward(retain_graph=True)
                self.SRoptimizers[j].step()
                sisr_loss.append(loss1.item())

            one_matrix = torch.ones(len(batch_ids),self.SR_COUNT).to(self.device)
            weight_identity = self.M[-1](one_matrix,labels)
            val,maxid = weight_identity.max(1) #have max of each row equal to 1
            loss3 = torch.mean(torch.abs(weight_identity[:,maxid] - 1))
            loss3.backward()
            self.O[-1].step()

        self.O[-1]
        self.M[-1]

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


