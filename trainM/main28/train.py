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
        self.TRAINING_LRPATH = glob.glob(os.path.join(args.training_lrpath,'x'+str(args.upsize),"*"))
        self.TRAINING_HRPATH = glob.glob(os.path.join(args.training_hrpath,'x'+str(args.upsize),"*"))
        self.TRAINING_LRPATH.sort()
        self.TRAINING_HRPATH.sort()
        self.PATCH_SIZE = args.patchsize
        self.TESTING_PATH = glob.glob(os.path.join(args.testing_path,'x'+str(args.upsize),"*"))
        self.LR = args.learning_rate
        self.upsize = args.upsize
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
            self.agent = agent.Agent(args)
            #self.agent = agent.Agent(args,chkpoint=loadedparams)
        else:
            self.agent = agent.Agent(args)
        self.SRmodels = []
        self.SRoptimizers = []
        self.schedulers = []
        for i in range(args.action_space):

            #CREATE THE ARCH
            if args.model == 'basic':
                model = arch.RRDBNet(3,3,32,args.d,gc=8,upsize=args.upsize)
            elif args.model == 'ESRGAN':
                model = arch.RRDBNet(3,3,64,23,gc=32,upsize=args.upsize)
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
                #model.load_state_dict(torch.load(args.ESRGAN_PATH),strict=True)
                loaded_dict = torch.load(args.ESRGAN_PATH)
                model_dict = model.state_dict()
                loaded_dict = {k: v for k, v in loaded_dict.items() if k in model_dict}
                model_dict.update(loaded_dict)
                model.load_state_dict(model_dict)
            elif args.model == 'RCAN':
                print('RCAN loaded!')
                model.load_state_dict(torch.load(args.pre_train,**kwargs),strict=True)
            elif args.model == 'basic':
                if args.d == 1:
                    model.load_state_dict(torch.load(args.basicpath_d1),strict=False)
                elif args.d == 2:
                    model.load_state_dict(torch.load(args.basicpath_d2),strict=True)
                elif args.d == 4:
                    model.load_state_dict(torch.load(args.basicpath_d4),strict=True)
                elif args.d == 8:
                    model.load_state_dict(torch.load(args.basicpath_d8),strict=True)
                else:
                    print('no pretrained model available. Random initialization of basic block')

            self.SRmodels.append(model)
            self.SRmodels[-1].to(self.device)

            self.SRoptimizers.append(torch.optim.Adam(model.parameters(),lr=1e-5))
            scheduler = torch.optim.lr_scheduler.StepLR(self.SRoptimizers[-1],200,gamma=0.8)

            self.schedulers.append(scheduler)

        #INCREMENT SCHEDULES TO THE CORRECT LOCATION
        #for i in range(args.step):
        #    [s.step() for s in self.schedulers]

    #TRAINING IMG LOADER WITH VARIABLE PATCH SIZES AND UPSCALE FACTOR
    def getTrainingPatches(self,LR,HR):
        patch_size = self.PATCH_SIZE
        stride = self.PATCH_SIZE

        #RANDOMLY FLIP AND ROTATE IMAGE
        bit1 = random.random() > 0.5
        bit2 = random.random() > 0.5
        bit3 = random.random() > 0.5
        if bit1:
            LR = np.rot90(LR)
            HR = np.rot90(HR)
        if bit2:
            LR = np.flip(LR,axis=1)
            HR = np.flip(HR,axis=1)
        if bit3:
            LR = np.flip(LR,axis=0)
            HR = np.flip(HR,axis=0)

        #ENSURE BOXES of size PATCH_SIZE CAN FIT OVER ENTIRE IMAGE
        h,w,d = LR.shape
        padh = patch_size - (h % patch_size)
        padw = patch_size - (w % patch_size)
        h = h + padh
        w = w + padw
        lrh, lrw = h,w
        LR = np.pad(LR,pad_width=((0,padh),(0,padw),(0,0)), mode='symmetric')       #symmetric padding to allow meaningful edges
        h,w,d = HR.shape
        padh = (patch_size*self.upsize) - (h % (patch_size*self.upsize))
        padw = (patch_size*self.upsize) - (w % (patch_size*self.upsize))
        h = h + padh
        w = w + padw
        hrh,hrw = h,w
        HR = np.pad(HR,pad_width=((0,padh),(0,padw),(0,0)),mode='symmetric')

        LR = torch.from_numpy(LR).float().unsqueeze(0)
        HR = torch.from_numpy(HR).float().unsqueeze(0)

        LR = LR.unfold(1,self.PATCH_SIZE,self.PATCH_SIZE).unfold(2,self.PATCH_SIZE,self.PATCH_SIZE).contiguous()
        HR = HR.unfold(1,self.PATCH_SIZE*self.upsize,self.PATCH_SIZE*self.upsize).unfold(2,self.PATCH_SIZE*self.upsize,self.PATCH_SIZE*self.upsize).contiguous()

        LR = LR.view(-1,3,self.PATCH_SIZE,self.PATCH_SIZE) / 255.0
        HR = HR.view(-1,3,self.PATCH_SIZE*self.upsize,self.PATCH_SIZE*self.upsize) / 255.0

        return LR,HR

    #SAVE THE AGENT AND THE SISR MODELS INTO A SINGLE FILE
    def savemodels(self):
            #sisrloss = torch.zeros(self.batch_size,self.PATCH_SIZE*self.UPSIZE,self.PATCH_SIZE*self.UPSIZE).to(self.device)
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

        unfold_LR = torch.nn.Unfold(kernel_size=self.PATCH_SIZE,stride=self.PATCH_SIZE,dilation=1)
        unfold_HR = torch.nn.Unfold(kernel_size=self.PATCH_SIZE*self.upsize,stride=self.PATCH_SIZE*self.upsize,dilation=1)

        #QUICK CHECK ON EVERYTHING
        with torch.no_grad():
            psnr,ssim,info = self.test.validateSet5(save=False,quick=False)

        #START TRAINING
        indices = list(range(len(self.TRAINING_HRPATH)))
        lossfn = torch.nn.L1Loss()
        lossMSE = torch.nn.MSELoss()
        lossCE = torch.nn.CrossEntropyLoss()
        softmaxfn = torch.nn.Softmax(dim=1)

        #random.shuffle(indices)
        for c in count():

            #FOR EACH HIGH RESOLUTION IMAGE
            for n,idx in enumerate(indices):
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
                    labels = torch.Tensor(batch_ids).long()
                    lrbatch = LR[labels,:,:,:]
                    hrbatch = HR[labels,:,:,:]
                    lrbatch = lrbatch.to(self.device)
                    hrbatch = hrbatch.to(self.device)

                    # GET SISR RESULTS FROM EACH MODEL
                    loss_SISR = 0
                    sisrs = []

                    # get probabilities
                    probs = self.agent.model(lrbatch)

                    for j, sisr in enumerate(self.SRmodels):
                        hr_pred = sisr(lrbatch)
                        sisrs.append(hr_pred)

                    #UPDATE BOTH THE SISR MODELS AND THE SELECTION MODEL ACCORDING TO THEIR LOSS
                    SR_result = torch.zeros(self.batch_size,3,self.PATCH_SIZE * self.upsize,self.PATCH_SIZE * self.upsize).to(self.device)
                    l1diff = []
                    for j, sr in enumerate(sisrs):
                        self.SRoptimizers[j].zero_grad()
                        diff = torch.abs(sr-hrbatch).sum(1)
                        #sisrloss += torch.mean(probs[:,j] * diff)
                        l1 = torch.abs(sr-hrbatch).mean(dim=1)
                        l1diff.append(l1)
                        pred = sr * probs[:,j].unsqueeze(1)
                        SR_result += pred
                    self.agent.opt.zero_grad()
                    maxval,maxidx = probs.max(dim=1)
                    l1diff = torch.stack(l1diff,dim=1)
                    minval,minidx = l1diff.min(dim=1)
                    #maxval2,maxidx2 = l1diff.max(dim=1)
                    sisrloss2 = lossfn(SR_result,hrbatch)
                    #sisrloss = minval.mean()                                           # min loss
                    sisrloss = torch.sum(l1diff.gather(1,maxidx.unsqueeze(1)))          # prob max loss
                    selectionloss = torch.mean(probs.gather(1,minidx.unsqueeze(1)).clamp(1e-16,1).log()) * -1       # cross entropy loss
                    #selectionloss2 = torch.mean(probs.gather(1,maxidx.unsqueeze(1)).clamp(1e-16,1).log())
                    #sisrloss_total = sisrloss2
                    #sisrloss_total = sisrloss + selectionloss
                    #sisrloss_total = (sisrloss + sisrloss2)*1000 + selectionloss
                    sisrloss_total = sisrloss2*1000 + selectionloss + sisrloss               # original
                    #sisrloss_total = selectionloss + minval.mean()
                    #sisrloss_total = selectionloss
                    #sisrloss_total = sisrloss + sisrloss2
                    sisrloss_total.backward()
                    [opt.step() for opt in self.SRoptimizers]
                    self.agent.opt.step()

                    diffmap = torch.nn.functional.softmax(-255 * (l1diff - torch.mean(l1diff)),dim=1)
                    reward = (l1diff - l1diff.mean(1).unsqueeze(1)).detach() * -1
                    reward = reward.sign()

                    target = torch.zeros(l1diff.shape)
                    for i in range(len(self.SRmodels)):
                        target[:,i] = (minidx == i).float()

                    #selectionloss = torch.mean(probs.gather(1,maxidx.unsqueeze(1)).clamp(1e-10,1).log() * reward.gather(1,maxidx.unsqueeze(1)))
                    #selectionloss = torch.mean(probs.gather(1,maxidx.unsqueeze(1)) * reward)
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
                    #hist_summaries = {'actions': probs[0].view(-1), "choices": choice[0].view(-1)}
                    #img_summaries = {'sr/mask': probs[0][:3], 'sr/sr': SR_result[0].clamp(0,1), 'sr/hr': hrbatch[0].clamp(0,1),'sr/targetmask': target[0][:3]}
                    img_summaries = {'sr/mask': probs[0][:3], 'sr/sr': SR_result[0].clamp(0,1),'sr/targetmask': target[0][:3], 'sr/diffmap': diffmap[0][:3]}
                    self.logger.scalar_summary(scalar_summaries)
                    #self.logger.hist_summary(hist_summaries)
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
                            #variance = torch.from_numpy(info['variance']).unsqueeze(0)
                            #print(f"max var: {torch.max(variance)}, min var: {torch.min(variance)}")
                            #variance = variance / torch.max(variance)
                            self.logger.image_summary({'Testing/Test Assignment':mask[:3], 'Testing/SR':srimg, 'Testing/HR': hrimg, 'Testing/upperboundmask': best_mask[:3]})
                        self.savemodels()
                    self.logger.incstep()

########################################################################################################
########################################################################################################
########################################################################################################
if __name__ == '__main__':

    sisr = SISR()
    sisr.train()


