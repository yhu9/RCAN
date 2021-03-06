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

#OUR END-TO-END MODEL TRAINER
class SISR():
    def __init__(self, args=args):

        #RANDOM MODEL INITIALIZATION FUNCTION
        def init_weights(m):
            if isinstance(m,torch.nn.Linear) or isinstance(m,torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)

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
        self.logger = logger.Logger(args.name,self.step+1)   #create our logger for tensorboard in log directory
        self.device = torch.device(args.device) #determine cpu/gpu

        #DEFAULT START OR START ON PREVIOUSLY TRAINED EPOCH
        self.load(args)

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
                    module = import_module('model.'+args.model.lower())
                    model = module.make_model(args).to(self.device)
                    kwargs = {}
                else: print('error loading RCAN model. QUITING'); quit();

            #LOAD THE WEIGHTS
            if args.model_dir != "":
                model.load_state_dict(loadedparams["sisr"+str(i)])
                print('continuing training')
            elif args.model == 'ESRGAN':
                model.load_state_dict(torch.load(args.ESRGAN_PATH),strict=True)
            elif args.model == 'RCAN':
                model.load_state_dict(torch.load(args.pre_train,**kwargs),strict=True)

            self.SRmodels.append(model)
            self.SRmodels[-1].to(self.device)
            self.SRoptimizers.append(torch.optim.Adam(model.parameters(),lr=1e-4))
            self.schedulers.append(torch.optim.lr_scheduler.StepLR(self.SRoptimizers[-1],10000,gamma=0.1))

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

    #MAIN FUNCTION WHICH GETS PATCH INFO GIVEN CURRENT TRAINING SET AND PATCHSIZE
    def genPatchInfo(self):
        data = []
        for idx in range(len(self.TRAINING_HRPATH)):
            HRpath = self.TRAINING_HRPATH[idx]
            LRpath = self.TRAINING_LRPATH[idx]
            LR = cv2.imread(LRpath,cv2.IMREAD_COLOR)
            HR = cv2.imread(HRpath,cv2.IMREAD_COLOR)
            LR,HR = self.getTrainingPatches(LR,HR)
            data.append(LR.shape[0])
            print(LR.shape)
        np.save(self.patchinfo_dir,np.array(data))
        print("num images", len(data))
        print("total patches", sum(data))

    #TRAINING REGIMEN
    def train(self,maxepoch=20,start=.01,end=0.00001):
        #EACH EPISODE TAKE ONE LR/HR PAIR WITH CORRESPONDING PATCHES
        #AND ATTEMPT TO SUPER RESOLVE EACH PATCH

        #requires pytorch 1.1.0+ which is not possible on the server
        #scheduler = torch.optim.lr_scheduler.CyclicLR(self.agent.optimizer,base_lr=0.0001,max_lr=0.1)

        #CREATE A TESTER to test at n iterations
        test = Tester(self.agent, self.SRmodels,testset=['Set5'])
        lossfn = torch.nn.CrossEntropyLoss()
        softmin_fn = torch.nn.Softmin(dim=1)
        softmax_fn = torch.nn.Softmax(dim=1)

        #START TRAINING
        for c in range(maxepoch):
            indices = list(range(len(self.TRAINING_HRPATH)))
            random.shuffle(indices)

            #FOR EACH HIGH RESOLUTION IMAGE
            for n,idx in enumerate(indices):
                temperature = end + (start - end) * math.exp(-1 * self.logger.step / 400) #exponential decay from start to end
                HRpath = self.TRAINING_HRPATH[idx]
                LRpath = self.TRAINING_LRPATH[idx]
                LR = imageio.imread(LRpath)
                HR = imageio.imread(HRpath)
                LR,HR = self.getTrainingPatches(LR,HR)

                #WE MUST GO THROUGH EVERY SINGLE PATCH IN RANDOM ORDER
                patch_ids = list(range(len(LR)))
                random.shuffle(patch_ids)
                P = []
                for _ in range(5):
                    batch_ids = random.sample(patch_ids,self.batch_size)    #TRAIN ON A SINGLE IMAGE

                    labels = torch.Tensor(batch_ids).long().cuda()
                    lrbatch = LR[labels,:,:,:]
                    hrbatch = HR[labels,:,:,:]

                    #update the agent once
                    #GET SISR RESULTS FROM EACH MODEL
                    sisr_loss = []
                    R = torch.zeros((self.batch_size,len(self.SRmodels)),requires_grad=False).to(self.device) #CONTAINER TO STORE SISR RESULTS
                    Sloss = [0] * len(self.SRmodels)
                    for j,sisr in enumerate(self.SRmodels):
                        hr_pred = sisr(lrbatch)

                        #update sisr model based on weighted l1 loss
                        l1diff = torch.abs(hr_pred - hrbatch).view(len(batch_ids),-1).mean(1)           #64x1 vector
                        R[:,j] = l1diff.squeeze(0)
                        Sloss[j] = torch.mean(l1diff)

                        self.SRoptimizers[j].zero_grad()           #zero our sisr gradients
                        probs = softmax_fn(self.agent.model(lrbatch))       #ANNEALING PROCESS USING TEMPERATURE ON SOFTMAX TO MOVE TOWARDS HARD ASSIGNMENT

                        weighted_imgscore = probs[:,j] * l1diff
                        loss1 = torch.mean(weighted_imgscore)
                        loss1.backward()
                        self.SRoptimizers[j].step()
                        sisr_loss.append(loss1.item())


                    #OPTIMIZE TO OUTPUT HIGHER PROBABILITY FOR MIN LOSS VALUE
                    self.agent.opt.zero_grad()    #zero our policy gradients
                    R = -(R - torch.mean(R,dim=1).unsqueeze(1))
                    R = softmax_fn(R * 1/ temperature)
                    probs = self.agent.model(lrbatch)
                    if '0.4' in torch.__version__: Agent_loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(probs,dim=1),R.detach())
                    else: Agent_loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(probs,dim=1),R.detach(),reduction='batchmean')
                    Agent_loss.backward()
                    self.agent.opt.step()
                    P.append(softmax_fn(probs).detach().cpu())

                #INCRIMENT SCHEDULERS
                for s in self.schedulers: s.step()
                self.agent.scheduler.step()

                #LOG THE INFORMATION
                print('\rEpoch/img: {}/{} | Agent Loss: {:.4f}, SISR Loss: {:.4f}, Temp: {:.6f}, S1: {:.4f},  S2: {:.4f}, S3: {:.4f}'\
                      .format(c,n,Agent_loss.item(),np.sum(sisr_loss),temperature, Sloss[0],Sloss[1],Sloss[2]),end="\n")
                self.logger.scalar_summary({'AgentLoss': Agent_loss, 'SISRLoss': torch.tensor(np.mean(sisr_loss)), "S1": Sloss[0], "S2": Sloss[1], "S3": Sloss[2]})
                if not '0.4' in torch.__version__: self.logger.hist_summary('actions',torch.stack(P).view(-1))
                self.logger.incstep()

                #save the model after 200 images total of 800 images
                if self.logger.step % 200 == 0:
                    with torch.no_grad():
                        psnr,ssim = test.validate(save=False)
                    [model.train() for model in self.SRmodels]
                    if self.logger: self.logger.scalar_summary({'Testing_PSNR': psnr, 'Testing_SSIM': ssim})
                    self.savemodels()

########################################################################################################
########################################################################################################
########################################################################################################
if __name__ == '__main__':

    sisr = SISR()
    if args.gen_patchinfo:
        sisr.genPatchInfo()
    else:
        sisr.train()


