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
            elif args.model == 'RCAN' or args.model == 'random':
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
                model.apply(init_weights)
            elif args.model == 'ESRGAN':
                model.load_state_dict(torch.load(args.ESRGAN_PATH),strict=True)
            elif args.model == 'RCAN':
                model.load_state_dict(torch.load(args.pre_train,**kwargs),strict=True)

            self.SRmodels.append(model)
            self.SRmodels[-1].to(self.device)
            self.SRoptimizers.append(torch.optim.Adam(model.parameters(),lr=1e-4))
            self.schedulers.append(torch.optim.lr_scheduler.StepLR(self.SRoptimizers[-1],500,gamma=0.1))

        #INCREMENT SCHEDULES TO THE CORRECT LOCATION
        for i in range(args.step):
            [s.step() for s in self.schedulers]

    #TRAINING IMG LOADER WITH VARIABLE PATCH SIZES AND UPSCALE FACTOR
    def getTrainingPatches(self,LR,HR):
        patch_size = self.PATCH_SIZE
        stride = self.PATCH_SIZE

        #ENSURE BOXES of size PATCH_SIZE CAN FIT OVER ENTIRE IMAGE
        h,w,d = LR.shape
        padh = patch_size - (h % patch_size)
        padw = patch_size - (w % patch_size)
        h = h + padh
        w = w + padw
        lrh, lrw = h,w
        LR = np.pad(LR,pad_width=((0,padh),(0,padw),(0,0)), mode='symmetric')       #symmetric padding to allow meaningful edges
        h,w,d = HR.shape
        padh = (patch_size*self.UPSIZE) - (h % (patch_size*self.UPSIZE))
        padw = (patch_size*self.UPSIZE) - (w % (patch_size*self.UPSIZE))
        h = h + padh
        w = w + padw
        hrh,hrw = h,w
        HR = np.pad(HR,pad_width=((0,padh),(0,padw),(0,0)),mode='symmetric')

        #GET PATCHES USING NUMPY'S VIEW AS WINDOW FUNCTION
        maxpatch_lr = (lrh // patch_size) * (lrw // patch_size)
        maxpatch_hr = (hrh // (patch_size*self.UPSIZE)) * (hrw // (patch_size*self.UPSIZE))
        LRpatches = view_as_windows(LR,(patch_size,patch_size,3),stride)
        HRpatches = view_as_windows(HR,(patch_size*self.UPSIZE,patch_size*self.UPSIZE,3),stride*4)

        #RESHAPE CORRECTLY AND CONVERT TO PYTORCH TENSOR
        LRpatches = torch.from_numpy(LRpatches).float()
        HRpatches = torch.from_numpy(HRpatches).float()
        LRpatches = LRpatches.permute(2,0,1,3,4,5).contiguous().view(-1,patch_size,patch_size,3)
        HRpatches = HRpatches.permute(2,0,1,3,4,5).contiguous().view(-1,patch_size*self.UPSIZE,patch_size*self.UPSIZE,3)
        LRpatches = LRpatches.permute(0,3,1,2)
        HRpatches = HRpatches.permute(0,3,1,2)

        if self.model == 'ESRGAN':
            LRpatches = LRpatches * 1.0 / 255
            HRpatches = HRpatches * 1.0 / 255

        return LRpatches.to(self.device),HRpatches.to(self.device)

    #APPLY SISR on a LR patch AND OPTIMIZE THAT PARTICULAR SISR MODEL ON CORRESPONDING HR PATCH
    def applySISR(self,lr,action,hr):

        self.SRoptimizers[action].zero_grad()
        hr_hat = self.SRmodels[action](lr)
        loss = F.l1_loss(hr_hat,hr)
        loss.backward()
        self.SRoptimizers[action].step()

        hr_hat = hr_hat.squeeze(0).permute(1,2,0); hr = hr.squeeze(0).permute(1,2,0)
        hr_hat = hr_hat.detach().cpu().numpy()
        hr = hr.detach().cpu().numpy()
        psnr,ssim = util.calc_metrics(hr_hat,hr,crop_border=self.UPSIZE)

        return hr_hat, psnr, ssim, loss.item()

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
    def train(self,maxepoch=20,start=.01,end=0.0001):
        #EACH EPISODE TAKE ONE LR/HR PAIR WITH CORRESPONDING PATCHES
        #AND ATTEMPT TO SUPER RESOLVE EACH PATCH

        #requires pytorch 1.1.0+ which is not possible on the server
        #scheduler = torch.optim.lr_scheduler.CyclicLR(self.agent.optimizer,base_lr=0.0001,max_lr=0.1)

        lossfn = torch.nn.CrossEntropyLoss()
        softmax_fn = torch.nn.Softmax(dim=1)

        #START TRAINING
        indices = list(range(len(self.TRAINING_HRPATH)))
        #random.shuffle(indices)
        for c in range(maxepoch):

            #FOR EACH HIGH RESOLUTION IMAGE
            for n,idx in enumerate(indices):
                #initialize temperature according to CURRENT NUMBER OF STEPS
                #temperature = end + (start - end) * math.exp(-1 * self.logger.step / 400)
                idx = random.sample(indices,1)[0]

                #GET INPUT FROM CURRENT IMAGE
                HRpath = self.TRAINING_HRPATH[idx]
                LRpath = self.TRAINING_LRPATH[idx]
                LR = imageio.imread(LRpath)
                HR = imageio.imread(HRpath)
                LR,HR = self.getTrainingPatches(LR,HR)

                #WE MUST GO THROUGH EVERY SINGLE PATCH IN RANDOM ORDER
                patch_ids = list(range(len(LR)))
                random.shuffle(patch_ids)
                P = []
                for step in range(1):
                    batch_ids = random.sample(patch_ids,self.batch_size)    #TRAIN ON A SINGLE IMAGE
                    labels = torch.Tensor(batch_ids).long().cuda()
                    lrbatch = LR[labels,:,:,:]
                    hrbatch = HR[labels,:,:,:]

                    #update the agent once
                    #GET SISR RESULTS FROM EACH MODEL
                    self.agent.opt.zero_grad()
                    SR_result = torch.zeros(self.batch_size,self.SR_COUNT,self.PATCH_SIZE * self.UPSIZE,self.PATCH_SIZE * self.UPSIZE).to(self.device)
                    Wloss = torch.zeros(self.batch_size,self.SR_COUNT,self.PATCH_SIZE * self.UPSIZE,self.PATCH_SIZE * self.UPSIZE).to(self.device)
                    loss_SISR = 0
                    probs = self.agent.model(lrbatch)
                    for j,sisr in enumerate(self.SRmodels):
                        self.SRoptimizers[j].zero_grad()           #zero our sisr gradients
                        hr_pred = sisr(lrbatch)
                        weighted_pred = hr_pred * probs[:,j].unsqueeze(1)
                        SR_result += weighted_pred

                    #CALCULATE LOSS
                    if self.model == 'ESRGAN':
                        l1diff = torch.mean(torch.abs(SR_result - hrbatch))
                    elif self.model == 'RCAN':
                        l1diff = torch.mean(torch.abs(hr_pred - hrbatch) / 255.0)
                    total_loss = l1diff
                    total_loss.backward()

                    #OPTIMIZE AND MOVE THE LEARNING RATE ACCORDING TO SCHEDULER
                    self.agent.opt.step()
                    [opt.step() for opt in self.SRoptimizers]
                    [sched.step() for sched in self.schedulers]
                    self.agent.scheduler.step()

                    #CONSOLE OUTPUT FOR QUICK AND DIRTY DEBUGGING
                    choice = probs.max(dim=1)[1]
                    c1 = (choice == 0).float().mean()
                    c2 = (choice == 1).float().mean()
                    c3 = (choice == 2).float().mean()
                    print('\rEpoch/img: {}/{} | Agent Loss: {:.4f}, SISR Loss: {:.4f}, c1: {:.4f}, c2: {:.4f}, c3: {:.4f}'\
                            .format(c,n,total_loss.item(),loss_SISR, c1.item(), c2.item(), c3.item()),end="\n")

                    #LOG AND SAVE THE INFORMATION
                    scalar_summaries = {'Loss/AgentLoss': total_loss, 'Loss/SISRLoss': loss_SISR, "choice/c1": c1, "choice/c2": c2, "choice/c3": c3}
                    hist_summaries = {'actions': probs[0].view(-1), "choices": choice[0].view(-1)}
                    if self.model == 'RCAN':
                        img_summaries = {'sr/mask': probs[0], 'sr/sr': (SR_result[0]/255.0).clamp(0,1), 'sr/hr': (hrbatch[0]/255.0).clamp(0,1)}
                    else:
                        img_summaries = {'sr/mask': probs[0], 'sr/sr': SR_result[0].clamp(0,1), 'sr/hr': hrbatch[0].clamp(0,1)}
                    self.logger.scalar_summary(scalar_summaries)
                    self.logger.hist_summary(hist_summaries)
                    self.logger.image_summary(img_summaries)
                    if self.logger.step % 100 == 0:
                        with torch.no_grad():
                            if not self.logger.step % 500 == 0:
                                psnr,ssim,info = self.test.validate(save=False,quick=True)
                            else:
                                psnr,ssim,info = self.test.validate(save=False,quick=False)
                        self.agent.model.train()
                        [model.train() for model in self.SRmodels]
                        if self.logger:
                            self.logger.scalar_summary({'Testing_PSNR': psnr, 'Testing_SSIM': ssim})
                            masked_sr = torch.from_numpy(info['assignment']).float().permute(2,0,1)
                            srimg = (torch.from_numpy(info['SRimg']).float()).permute(2,0,1)
                            hrimg = (torch.from_numpy(info['HRimg']).float()).permute(2,0,1)
                            hrimg = hrimg / 255.0
                            srimg = srimg / 255.0
                            self.logger.image_summary({'Testing/Test Assignment':masked_sr, 'Testing/SR':srimg, 'Testing/HR': hrimg})
                        self.savemodels()
                    self.logger.incstep()

########################################################################################################
########################################################################################################
########################################################################################################
if __name__ == '__main__':

    sisr = SISR()
    sisr.train()


