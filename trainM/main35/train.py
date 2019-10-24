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

    #SAVE THE AGENT AND THE SISR MODELS INTO A SINGLE FILE
    def savemodels(self):
        data = {}
        data['agent'] = self.agent.model.state_dict()
        for i,m in enumerate(self.SRmodels):
            modelname = "sisr" + str(i)
            data[modelname] = m.state_dict()
        data['step'] = self.logger.step
        torch.save(data,"models/" + self.name + "_sisr.pth")

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

    # evaluate the input and get sr result
    def getGroundTruthIOU(self,lr,hr,samplesize=0.1):
        if self.model == 'ESRGAN':
            lr = lr / 255.0
            hr = hr / 255.0

        lr = torch.FloatTensor(lr).to(self.device)
        hr = torch.FloatTensor(hr).to(self.device)
        lr = lr.permute((2,0,1)).unsqueeze(0)
        hr = hr.permute((2,0,1)).unsqueeze(0)

        # GATHER BEST RESULTS AND BEST GUESS
        choice = self.agent.model(lr)
        maxchoice, idxguess = choice.max(dim=1)
        l2diff = []
        for i, sisr in enumerate(self.SRmodels):
            sr = sisr(lr)
            pixelMSE = (sr - hr).pow(2).sqrt().mean(dim=1)
            l2diff.append(pixelMSE)
        l2diff = torch.stack(l2diff)
        minvals, idxbest = l2diff.min(dim=0)
        correct = idxguess == idxbest
        iou = correct.float().sum() / (idxbest.shape[1] * idxbest.shape[2])
        return iou.cpu().item()


    # GATHER INFO ABOUT DIFFICULTY OF data
    def getDifficulty(self,data):
        [sisr.eval() for sisr in self.SRmodels]
        self.agent.model.eval()
        difficulty = []
        for d in data:
            lrpath = self.TRAINING_LRPATH[d]
            hrpath = self.TRAINING_HRPATH[d]
            LR = imageio.imread(lrpath)
            HR = imageio.imread(hrpath)

            with torch.no_grad():
                iou = self.getGroundTruthIOU(LR,HR)

            difficulty.append( (d,iou) )
        difficulty = sorted(difficulty, key=lambda x:x[1])
        return difficulty

    # optimization function for both the superresolution models and the selection network
    # input:
    #   data -> list of int
    #   maxepoch -> int (default=5)
    # output:
    #   none
    #
    # description:
    #   optimizes the super resolution model according to the list of indices passed in as input
    #   for data. indices represent unique low resolution and high resolution pairs
    #
    def optimize(self,data,maxepoch=5):
        #EACH EPISODE TAKE ONE LR/HR PAIR WITH CORRESPONDING PATCHES
        #AND ATTEMPT TO SUPER RESOLVE EACH PATCH

        #QUICK CHECK ON EVERYTHING
        #with torch.no_grad():
        #    psnr,ssim,info = self.test.validateSet5(save=False,quick=False)

        #START TRAINING
        lossfn = torch.nn.L1Loss()

        random.shuffle(data)
        for c in range(maxepoch):
            for n,idx in enumerate(data):
                idx = random.sample(data,1)[0]

                #GET INPUT FROM CURRENT IMAGE
                HRpath = self.TRAINING_HRPATH[idx]
                LRpath = self.TRAINING_LRPATH[idx]
                LR = imageio.imread(LRpath)
                HR = imageio.imread(HRpath)

                LR,HR = util.getTrainingPatches(LR,HR,args)

                #WE MUST GO THROUGH EVERY SINGLE PATCH IN RANDOM ORDER
                patch_ids = list(range(len(LR)))
                random.shuffle(patch_ids)
                for step in range(1):
                    batch_ids = random.sample(patch_ids,self.batch_size)    #TRAIN ON A SINGLE IMAGE
                    labels = torch.Tensor(batch_ids).long()
                    lr_batch = LR[labels,:,:,:]
                    hr_batch = HR[labels,:,:,:]
                    lr_batch = lr_batch.to(self.device)
                    hr_batch = hr_batch.to(self.device)
                    if args.model == 'ESRGAN':
                        lr_batch = lr_batch / 255.0
                        hr_batch = hr_batch / 255.0

                    #GET SISR RESULTS FROM EACH MODEL
                    sisrs = []
                    probs = self.agent.model(lr_batch)
                    for j, sisr in enumerate(self.SRmodels):
                        hr_pred = sisr(lr_batch)
                        sisrs.append(hr_pred)

                    #UPDATE BOTH THE SISR MODELS AND THE SELECTION MODEL ACCORDING TO THEIR LOSS
                    SR_result = torch.zeros(self.batch_size,3,self.PATCH_SIZE * self.UPSIZE,self.PATCH_SIZE * self.UPSIZE).to(self.device)
                    SR_result.requires_grad = False
                    sisrloss = 0
                    l1diff = []
                    for j, sr in enumerate(sisrs):
                        self.SRoptimizers[j].zero_grad()
                        loss = torch.mean(lossfn(sr,hr_batch) * probs[:,j].unsqueeze(1))
                        sisrloss += loss
                        l1 = torch.abs(sr-hr_batch).mean(dim=1)
                        l1diff.append(l1)
                        pred = sr * probs[:,j].unsqueeze(1)
                        SR_result += pred
                    self.agent.opt.zero_grad()
                    maxval,maxidx = probs.max(dim=1)
                    l1diff = torch.stack(l1diff,dim=1)
                    #sisrloss = lossfn(SR_result,hr_batch)
                    sisrloss_total = sisrloss
                    sisrloss_total.backward()
                    [opt.step() for opt in self.SRoptimizers]
                    self.agent.opt.step()

                    diffmap = torch.nn.functional.softmax(-255 * (l1diff - torch.mean(l1diff)),dim=1)
                    minval,minidx = l1diff.min(dim=1)
                    target = torch.nn.functional.one_hot(minidx,len(sisrs)).permute(0,3,1,2)    #TARGET PROBABILITY MASK WE HOPE FOR?
                    reward = (l1diff - l1diff.mean(1).unsqueeze(1)).detach() * -1
                    reward = (reward - reward.mean())/ reward.std()
                    #reward = reward.sign()
                    #selectionloss = torch.mean(probs.gather(1,maxidx.unsqueeze(1)).clamp(1e-10,1).log() * reward.gather(1,maxidx.unsqueeze(1)))
                    selectionloss = torch.mean(probs.gather(1,maxidx.unsqueeze(1)) * reward)

                    #UPDATE OUR AGENT
                    #self.agent.opt.zero_grad()
                    #probs = self.agent.model(lr_batch)
                    #maxval,maxidx = probs.max(dim=1)
                    #selectionloss = torch.mean(probs.gather(1,maxidx.unsqueeze(1)).clamp(1e-10,1).log() * reward.gather(1,maxidx.unsqueeze(1)))
                    #selectionloss.backward()
                    #self.agent.opt.step()

                    #[sched.step() for sched in self.schedulers]
                    #self.agent.scheduler.step()

                    #CONSOLE OUTPUT FOR QUICK AND DIRTY DEBUGGING
                    lr = self.SRoptimizers[-1].param_groups[0]['lr']
                    choice = probs.max(dim=1)[1]
                    c1 = (choice == 0).float().mean()
                    c2 = (choice == 1).float().mean()
                    c3 = (choice == 2).float().mean()
                    print('\rEpoch/img: {}/{} | LR: {:.8f} | Agent Loss: {:.4f}, SISR Loss: {:.4f}, c1: {:.4f}, c2: {:.4f}, c3: {:.4f}'\
                            .format(c,n,lr,selectionloss.item(),sisrloss.item(), c1.item(), c2.item(), c3.item()),end="\n")

                    #LOG AND SAVE THE INFORMATION
                    scalar_summaries = {'Loss/AgentLoss': selectionloss, 'Loss/SISRLoss': sisrloss, "choice/c1": c1, "choice/c2": c2, "choice/c3": c3}
                    hist_summaries = {'actions': probs[0].view(-1), "choices": choice[0].view(-1)}
                    img_summaries = {'sr/mask': probs[0][:3], 'sr/sr': SR_result[0].clamp(0,1),'sr/targetmask': target[0][:3], 'sr/diffmap': diffmap[0][:3]}
                    self.logger.scalar_summary(scalar_summaries)
                    self.logger.hist_summary(hist_summaries)
                    self.logger.image_summary(img_summaries)
                    if self.logger.step % 100 == 0:
                        with torch.no_grad():
                            psnr,ssim,info = self.test.validateSet5(save=False,quick=False)
                        self.agent.model.train()
                        [model.train() for model in self.SRmodels]
                        if self.logger:
                            self.logger.scalar_summary({'Testing_PSNR': psnr, 'Testing_SSIM': ssim})
                            mask = torch.from_numpy(info['choices']).float().permute(2,0,1)
                            best_mask = info['upperboundmask'].squeeze()
                            worst_mask = info['lowerboundmask'].squeeze()
                            hrimg = info['HR'].squeeze()
                            srimg = torch.from_numpy(info['weighted']).permute(2,0,1).clamp(0,1)
                            variance = torch.from_numpy(info['variance']).unsqueeze(0)
                            print(f"max var: {torch.max(variance)}, min var: {torch.min(variance)}")
                            variance = variance / torch.max(variance)
                            advantage = info['advantage'].squeeze(1)
                            self.logger.image_summary({'Testing/Test Assignment':mask[:3], 'Testing/SR':srimg, 'Testing/HR': hrimg, 'Testing/upperboundmask': best_mask, 'Testing/var': variance, 'Testing/advantage':advantage})
                        self.savemodels()
                    self.logger.incstep()

    # TRAINING REGIMEN
    def train(self,alpha=0, beta=3):
        data = set(range(len(self.TRAINING_HRPATH)))
        data.remove(alpha)
        curriculum = [alpha]

        self.optimize(curriculum,maxepoch=1000)

        # main training loop
        for i in count():
            difficulty = self.getDifficulty(data)
            np.save('curriculum_' + str(i),np.array(difficulty))
            A = [a[0] for a in difficulty[-beta:]]
            curriculum += A
            [data.remove(a) for a in A]
            self.optimize(curriculum)

########################################################################################################
########################################################################################################
########################################################################################################
if __name__ == '__main__':

    sisr = SISR()
    sisr.train()


