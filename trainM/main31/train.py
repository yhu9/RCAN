#NATIVE IMPORTS
import os
import glob
import argparse
from collections import deque
from itertools import count
import random
import time
import imageio
import math

#OPEN SOURCE IMPORTS
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as transforms
from sklearn.feature_extraction import image
from skimage.util.shape import view_as_windows

#CUSTOM IMPORTS
import RRDBNet_arch as arch
import agent
import logger
import utility
import model
from option import args     #COMMAND LINE ARGUMENTS VIEW option.py file
from importlib import import_module
from utils import util
from test import Tester

torch.manual_seed(args.seed)
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

#RANDOM MODEL INITIALIZATION FUNCTION
def init_zero(m):
    if isinstance(m,torch.nn.Linear) or isinstance(m,torch.nn.Conv2d):
        m.weight.data.fill_(0.00)
        m.bias.data.fill_(0.00)

def init_weights(m):
    if isinstance(m,torch.nn.Linear) or isinstance(m,torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0.01)

#OUR END-TO-END MODEL TRAINER
class SISR():
    def __init__(self, args=args):

        #INITIALIZE VARIABLES
        self.SR_COUNT = args.action_space
        SRMODEL_PATH = args.srmodel_path
        self.batch_size = args.batch_size
        self.TRAINING_LRPATH = glob.glob(os.path.join(args.training_lrpath,"*"))
        self.TRAINING_HRPATH = glob.glob(os.path.join(args.training_hrpath,"*"))
        self.TRAINING_LRPATH.sort()
        self.TRAINING_HRPATH.sort()
        self.PATCH_SIZE = args.patchsize
        self.TESTING_PATH = glob.glob(os.path.join(args.testing_path,"*"))
        self.LR = args.learning_rate
        self.UPSIZE = args.upsize
        self.step = args.step
        self.name = args.name
        self.logger = logger.Logger(args.name,self.step)   #create our logger for tensorboard in log directory
        self.device = torch.device(args.device) #determine cpu/gpu
        self.model = args.model

        #DEFAULT START OR START ON PREVIOUSLY TRAINED EPOCH
        self.load(args)

        #INITIALIZE TESTING MODULE
        self.test = Tester(self.agent, self.SRmodels,args=args,testset=['Set5'])

    #LOAD A PRETRAINED AGENT WITH SUPER RESOLUTION MODELS
    def load(self,args):

        if args.model_dir != "":
            loadedparams = torch.load(args.model_dir,map_location=self.device)
            self.agent = agent.Agent(args,chkpoint=loadedparams)
        else:
            self.agent = agent.Agent(args)
        self.SRmodels = []
        self.SRoptimizers = []
        self.schedulers = []
        for i in range(args.action_space):

            #CREATE THE ARCH
            if args.model == 'ESRGAN':
                model = arch.RRDBNet(3,3,64,23,gc=32)
            elif args.model == 'RCAN':
                torch.manual_seed(args.seed)
                checkpoint = utility.checkpoint(args)
                if checkpoint.ok:
                    module = import_module('model.rcan')
                    model = module.make_model(args).to(self.device)
                    kwargs = {}
                else: print('error loading RCAN model. QUITING'); quit();

            #LOAD THE WEIGHTS
            if args.model_dir != "":
                model.load_state_dict(loadedparams["sisr"+str(i)])
                print('continuing training')
            elif args.random:
                print('random init')
            elif args.model == 'ESRGAN':
                model.load_state_dict(torch.load(args.ESRGAN_PATH),strict=True)
            elif args.model == 'RCAN':
                print('RCAN loaded!')
                model.load_state_dict(torch.load(args.pre_train,**kwargs),strict=True)

            self.SRmodels.append(model)
            self.SRmodels[-1].to(self.device)

            self.SRoptimizers.append(torch.optim.Adam(model.parameters(),lr=1e-5))
            scheduler = torch.optim.lr_scheduler.StepLR(self.SRoptimizers[-1],200,gamma=0.8)

            self.schedulers.append(scheduler)

        #INCREMENT SCHEDULES TO THE CORRECT LOCATION
        for i in range(args.step):
            [s.step() for s in self.schedulers]

    #SAVE THE AGENT AND THE SISR MODELS INTO A SINGLE FILE
    def savemodels(self):
        data = {}
        data['agent'] = self.agent.model.state_dict()
        for i,m in enumerate(self.SRmodels):
            modelname = "sisr" + str(i)
            data[modelname] = m.state_dict()
        data['step'] = self.logger.step
        torch.save(data,"models/" + self.name + "_sisr.pth")

    #TRAINING REGIMEN
    def train(self,maxepoch=100,start=.01,end=0.0001):
        #EACH EPISODE TAKE ONE LR/HR PAIR WITH CORRESPONDING PATCHES
        #AND ATTEMPT TO SUPER RESOLVE EACH PATCH

        #requires pytorch 1.1.0+ which is not possible on the server
        #scheduler = torch.optim.lr_scheduler.CyclicLR(self.agent.optimizer,base_lr=0.0001,max_lr=0.1)
        unfold_LR = torch.nn.Unfold(kernel_size=self.PATCH_SIZE,stride=self.PATCH_SIZE,dilation=1)
        unfold_HR = torch.nn.Unfold(kernel_size=self.PATCH_SIZE*4,stride=self.PATCH_SIZE*4,dilation=1)

        #QUICK CHECK ON EVERYTHING
        #with torch.no_grad():
        #    psnr,ssim,info = self.test.validateSet5(save=False,quick=False)

        #START TRAINING
        indices = list(range(len(self.TRAINING_HRPATH)))
        lossfn = torch.nn.L1Loss()
        lossMSE = torch.nn.MSELoss()
        lossCE = torch.nn.CrossEntropyLoss()
        softmaxfn = torch.nn.Softmax(dim=1)

        #random.shuffle(indices)
        for c in range(maxepoch):

            #FOR EACH HIGH RESOLUTION IMAGE
            for n,idx in enumerate(indices):
                idx = random.sample(indices,1)[0]
                #idx = 0

                #GET INPUT FROM CURRENT IMAGE
                HRpath = self.TRAINING_HRPATH[idx]
                LRpath = self.TRAINING_LRPATH[idx]
                LR = imageio.imread(LRpath)
                HR = imageio.imread(HRpath)

                LR,HR,_ = util.getTrainingPatches(LR,HR,args)

                # WE GO THROUGH PATCH IN RANDOM ORDER
                patch_ids = list(range(len(LR)))
                random.shuffle(patch_ids)
                P = []
                for step in range(1):
                    batch_ids = random.sample(patch_ids,self.batch_size)
                    #batch_ids = patch_ids[:self.batch_size]
                    labels = torch.Tensor(batch_ids).long()

                    lrbatch = LR[labels,:,:,:]
                    hrbatch = HR[labels,:,:,:]

                    lrbatch = lrbatch.to(self.device)
                    hrbatch = hrbatch.to(self.device)

                    #GET SISR RESULTS FROM EACH MODEL
                    loss_SISR = 0
                    sisrs = []
                    probs = self.agent.model(lrbatch)
                    for sisr in self.SRmodels:
                        hr_pred = sisr(lrbatch)
                        sisrs.append(hr_pred)

                    #UPDATE BOTH THE SISR MODELS AND THE SELECTION MODEL ACCORDING TO THEIR LOSS
                    sisrloss = []
                    l1loss = []
                    for j, sr in enumerate(sisrs):
                        self.SRoptimizers[j].zero_grad()
                        l1 = torch.abs(sr - hrbatch).sum(dim=1).sum(dim=1).sum(dim=1) / ((self.PATCH_SIZE * self.UPSIZE)**2 * 3)
                        l1loss.append(l1)
                        loss = torch.mean(l1 * probs[:,j])
                        sisrloss.append(loss)

                    l1loss = torch.stack(l1loss,dim=1)
                    self.agent.opt.zero_grad()
                    sisrloss_total = sum(sisrloss)
                    sisrloss_total.backward()
                    [opt.step() for opt in self.SRoptimizers]
                    self.agent.opt.step()

                    #[sched.step() for sched in self.schedulers]
                    #self.agent.scheduler.step()

                    #CONSOLE OUTPUT FOR QUICK AND DIRTY DEBUGGING
                    lr = self.SRoptimizers[-1].param_groups[0]['lr']
                    lr2 = self.agent.opt.param_groups[0]['lr']
                    _,maxarg = probs[0].max(0)
                    sampleSR = sisrs[maxarg.item()][0] / 255.0
                    sampleHR = hrbatch[0] / 255.0
                    choice = probs.max(dim=1)[1]
                    c1 = (choice == 0).float().mean()
                    c2 = (choice == 1).float().mean()
                    c3 = (choice == 2).float().mean()
                    s1 = sisrloss[0]
                    s2 = sisrloss[1]
                    s3 = sisrloss[2]
                    #s1 = torch.mean(l1loss[:,0])
                    #s2 = torch.mean(l1loss[:,1])
                    #s3 = torch.mean(l1loss[:,2])
                    agentloss = torch.mean(l1loss.gather(1,choice.unsqueeze(1)))
                    print('\rEpoch/img: {}/{} | LR sr/ag: {:.8f}/{:.8f} | Agent Loss: {:.4f}, SISR Loss: {:.4f} | s1: {:.4f} | s2: {:.4f} | s3: {:.4f}'\
                            .format(c,n,lr,lr2,agentloss.item(),sisrloss_total.item(),s1.item(),s2.item(),s3.item()),end="\n")

                    #LOG AND SAVE THE INFORMATION
                    scalar_summaries = {'Loss/AgentLoss': agentloss, 'Loss/SISRLoss': sisrloss_total, "choice/c1": c1, "choice/c2": c2, "choice/c3": c3, "sisr/s1": s1, "sisr/s2": s2, "sisr/s3": s3}
                    hist_summaries = {'actions': probs.view(-1), "choices": choice.view(-1)}
                    img_summaries = {'sr/HR': sampleHR.clamp(0,1),'sr/SR': sampleSR.clamp(0,1)}
                    self.logger.hist_summary(hist_summaries)
                    self.logger.scalar_summary(scalar_summaries)
                    self.logger.image_summary(img_summaries)
                    if self.logger.step % 100 == 0:
                        with torch.no_grad():
                            psnr,ssim,info = self.test.validateSet5(save=False,quick=False)
                        self.agent.model.train()
                        [model.train() for model in self.SRmodels]
                        if self.logger:
                            self.logger.scalar_summary({'Testing_PSNR': psnr, 'Testing_SSIM': ssim})
                            weightedmask = torch.from_numpy(info['mask']).permute(2,0,1) / 255.0
                            mask = torch.from_numpy(info['maxchoice']).permute(2,0,1) / 255.0
                            optimal_mask = torch.from_numpy(info['optimalchoice']).permute(2,0,1) / 255.0

                            hrimg = torch.Tensor(info['HR']).permute(2,0,1) / 255.0
                            srimg = torch.from_numpy(info['max']).permute(2,0,1) / 255.0
                            self.logger.image_summary({'Testing/Test Assignment':mask[:3],'Testing/Weight': weightedmask[:3], 'Testing/SR':srimg, 'Testing/HR': hrimg, 'Testing/optimalmask': optimal_mask})
                        #self.savemodels()
                    self.logger.incstep()

########################################################################################################
########################################################################################################
########################################################################################################
if __name__ == '__main__':

    sisr = SISR()
    sisr.train()


