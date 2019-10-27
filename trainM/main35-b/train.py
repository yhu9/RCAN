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
            if args.model == 'basic':
                model = arch.RRDBNet(3,3,32,1,gc=8)
            elif args.model == 'ESRGAN':
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
            elif args.model == 'basic':
                model.apply(init_weights)

            self.SRmodels.append(model)
            self.SRmodels[-1].to(self.device)

            self.SRoptimizers.append(torch.optim.Adam(model.parameters(),lr=1e-5))
            scheduler = torch.optim.lr_scheduler.StepLR(self.SRoptimizers[-1],200,gamma=0.8)

            self.schedulers.append(scheduler)

        #INCREMENT SCHEDULES TO THE CORRECT LOCATION
        for i in range(args.step):
            [s.step() for s in self.schedulers]

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
        padh = (patch_size*self.UPSIZE) - (h % (patch_size*self.UPSIZE))
        padw = (patch_size*self.UPSIZE) - (w % (patch_size*self.UPSIZE))
        h = h + padh
        w = w + padw
        hrh,hrw = h,w
        HR = np.pad(HR,pad_width=((0,padh),(0,padw),(0,0)),mode='symmetric')

        LR = torch.from_numpy(LR).float().unsqueeze(0)
        HR = torch.from_numpy(HR).float().unsqueeze(0)

        LR = LR.unfold(1,self.PATCH_SIZE,self.PATCH_SIZE).unfold(2,self.PATCH_SIZE,self.PATCH_SIZE).contiguous()
        HR = HR.unfold(1,self.PATCH_SIZE*self.UPSIZE,self.PATCH_SIZE*self.UPSIZE).unfold(2,self.PATCH_SIZE*self.UPSIZE,self.PATCH_SIZE*self.UPSIZE).contiguous()

        LR = LR.view(-1,3,self.PATCH_SIZE,self.PATCH_SIZE)
        HR = HR.view(-1,3,self.PATCH_SIZE*self.UPSIZE,self.PATCH_SIZE*self.UPSIZE)

        return LR,HR

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
    def optimize(self,data,iou_threshold=0.8):
        #EACH EPISODE TAKE ONE LR/HR PAIR WITH CORRESPONDING PATCHES
        #AND ATTEMPT TO SUPER RESOLVE EACH PATCH

        #QUICK CHECK ON EVERYTHING
        with torch.no_grad():
            psnr,ssim,info = self.test.validateSet5(save=False,quick=False)

        agent_iou = deque(maxlen=100)
        while True:

            #GET INPUT FROM CURRENT IMAGE
            idx = 0
            HRpath = self.TRAINING_HRPATH[idx]
            LRpath = self.TRAINING_LRPATH[idx]
            LR = imageio.imread(LRpath)
            HR = imageio.imread(HRpath)
            LR,HR = self.getTrainingPatches(LR,HR)
            patch_ids = list(range(len(LR)))

            batch_ids = random.sample(patch_ids,self.batch_size)
            labels = torch.Tensor(batch_ids).long()

            lrbatch = LR[labels,:,:,:]
            hrbatch = HR[labels,:,:,:]
            lrbatch = lrbatch.to(self.device)
            hrbatch = hrbatch.to(self.device)

            #GET SISR RESULTS FROM EACH MODEL
            sisrs = []
            probs = self.agent.model(lrbatch)
            for j, sisr in enumerate(self.SRmodels):
                hr_pred = sisr(lrbatch)
                sisrs.append(hr_pred)

            #UPDATE BOTH THE SISR MODELS AND THE SELECTION MODEL ACCORDING TO THEIR LOSS
            SR_result = torch.zeros(self.batch_size,3,self.PATCH_SIZE * self.UPSIZE,self.PATCH_SIZE * self.UPSIZE).to(self.device)
            SR_result.requires_grad = False
            l1diff = []
            sisrloss = 0
            for j, sr in enumerate(sisrs):
                self.SRoptimizers[j].zero_grad()
                l1 = torch.abs(sr-hrbatch).mean(dim=1)
                sisrloss += torch.mean(probs[:,j] * l1)
                l1diff.append(l1)
                pred = sr * probs[:,j].unsqueeze(1)
                SR_result += pred
            self.agent.opt.zero_grad()
            maxval,maxidx = probs.max(dim=1)
            l1diff = torch.stack(l1diff,dim=1)
            sisrloss_total = sisrloss
            sisrloss_total.backward()
            [opt.step() for opt in self.SRoptimizers]
            self.agent.opt.step()

            diffmap = torch.nn.functional.softmax(-1 * (l1diff - torch.mean(l1diff)),dim=1)
            minval,minidx = l1diff.min(dim=1)
            reward = (l1diff - l1diff.mean(1).unsqueeze(1)).detach() * -1
            reward = reward.sign()
            target = torch.nn.functional.one_hot(minidx,len(sisrs)).permute(0,3,1,2)    #TARGET PROBABILITY MASK WE HOPE FOR?
            selectionloss = torch.mean(probs.gather(1,maxidx.unsqueeze(1)).clamp(1e-10,1).log() * reward.gather(1,maxidx.unsqueeze(1)))

            #CONSOLE OUTPUT FOR QUICK AND DIRTY DEBUGGING
            lr = self.SRoptimizers[-1].param_groups[0]['lr']
            lr2 = self.agent.opt.param_groups[0]['lr']
            SR_result = SR_result / 255
            hrbatch = hrbatch / 255
            choice = probs.max(dim=1)[1]
            iou = (choice == minidx).float().sum() / (choice.shape[0] * choice.shape[1] * choice.shape[2])
            agent_iou.append(iou.item())
            c1 = (choice == 0).float().mean()
            c2 = (choice == 1).float().mean()
            c3 = (choice == 2).float().mean()
            s1 = torch.mean(l1diff[0])
            s2 = torch.mean(l1diff[1])
            s3 = torch.mean(l1diff[2])
            print('\rEpoch/img: {} | LR sr/ag: {:.8f}/{:.8f} | Agent Loss: {:.4f}, SISR Loss: {:.4f}, | IOU: {:.4f} | s1: {:.4f}, s2: {:.4f}, s3: {:.4f}'\
                    .format(self.logger.step,lr,lr2,selectionloss.item(),sisrloss.item(), iou.item(), s1.item(), s2.item(), s3.item()),end="\n")

            #LOG AND SAVE THE INFORMATION
            scalar_summaries = {'Loss/AgentLoss': selectionloss, 'Loss/SISRLoss': sisrloss,'Loss/IOU': iou, "choice/c1": c1, "choice/c2": c2, "choice/c3": c3, "sisr/s1": s1, "sisr/s2": s2, "sisr/s3": s3}
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
                    mask = torch.from_numpy(info['choices']).float().permute(2,0,1) / 255.0
                    best_mask = info['upperboundmask'].squeeze()
                    worst_mask = info['lowerboundmask'].squeeze()
                    hrimg = info['HR'].squeeze() / 255.0
                    srimg = torch.from_numpy(info['weighted'] / 255.0).permute(2,0,1)
                    variance = torch.from_numpy(info['variance']).unsqueeze(0)
                    print(f"max var: {torch.max(variance)}, min var: {torch.min(variance)}")
                    variance = variance / torch.max(variance)
                    self.logger.image_summary({'Testing/Test Assignment':mask[:3], 'Testing/SR':srimg, 'Testing/HR': hrimg, 'Testing/upperboundmask': best_mask, 'Testing/var': variance})
                self.savemodels()
            self.logger.incstep()
            if np.mean(agent_iou) >= iou_threshold: break

    # train just a single sr model
    def train_basic(self, maxepoch=50):

        data = set(range(len(self.TRAINING_HRPATH)))
        for e in count():
            for c in range(len(self.TRAINING_HRPATH)):

                #GET INPUT FROM CURRENT IMAGE
                idx = random.sample(data,1)[0]
                HRpath = self.TRAINING_HRPATH[idx]
                LRpath = self.TRAINING_LRPATH[idx]
                LR = imageio.imread(LRpath)
                HR = imageio.imread(HRpath)
                LR,HR = self.getTrainingPatches(LR,HR)
                patch_ids = list(range(len(LR)))

                batch_ids = random.sample(patch_ids,self.batch_size)
                labels = torch.Tensor(batch_ids).long()

                lrbatch = LR[labels,:,:,:]
                hrbatch = HR[labels,:,:,:]
                lrbatch = lrbatch.to(self.device) / 255.0
                hrbatch = hrbatch.to(self.device) / 255.0

                self.SRoptimizers[-1].zero_grad()
                sr = self.SRmodels[-1](lrbatch)
                sisrloss = torch.abs(sr-hrbatch).mean()
                sisrloss.backward()
                self.SRoptimizers[-1].step()

                print('\rEpoch/img: {}/{}| SISR Loss: {:.4f}'\
                        .format(e,c,sisrloss.item()),end="\n")

                #LOG AND SAVE THE INFORMATION
                scalar_summaries = {'Loss/L1_basic': sisrloss}
                self.logger.scalar_summary(scalar_summaries)
                self.logger.incstep()
            psnr,ssim = self.test.testbasic()
            self.logger.scalar_summary({'test/psnr': psnr, 'test/ssim': ssim})
            torch.save(self.SRmodels[-1].state_dict(),"models/sisrbasic.pth")

########################################################################################################
########################################################################################################
########################################################################################################
if __name__ == '__main__':

    sisr = SISR()
    if args.model == 'basic':
        sisr.train_basic()
    else:
        sisr.train()


