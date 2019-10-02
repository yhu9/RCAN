#NATIVE IMPORTS
import os
import glob
import argparse
from collections import deque
from itertools import count
import random
import time
import imageio

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


        #INITIALIZE VARIABLES
        self.SR_COUNT = args.action_space
        SRMODEL_PATH = args.srmodel_path
        self.batch_size = args.batch_size
        self.TRAINING_LRPATH = glob.glob(os.path.join(args.training_lrpath,"*"))
        self.TRAINING_HRPATH = glob.glob(os.path.join(args.training_hrpath,"*"))
        self.TRAINING_LRPATH.sort()
        self.TRAINING_HRPATH.sort()
        self.PATCH_SIZE = args.patchsize
        self.patchinfo_dir = args.patchinfo
        self.TESTING_PATH = glob.glob(os.path.join(args.testing_path,"*"))
        self.LR = args.learning_rate
        self.UPSIZE = args.upsize
        self.step = 0
        self.name = args.name
        if args.name != 'none':
            self.logger = logger.Logger(args.name)   #create our logger for tensorboard in log directory
        else: self.logger = None
        self.device = torch.device(args.device) #determine cpu/gpu

        #DEFAULT START OR START ON PREVIOUSLY TRAINED EPOCH
        if args.model_dir != "":
            self.load(args)
            print('continue training for model: ' + args.model_dir)
        else:
            self.SRmodels = []
            self.SRoptimizers = []
            #LOAD A COPY OF THE MODEL N TIMES
            for i in range(self.SR_COUNT):
                model = self.load(args)
                self.SRmodels.append(model)
                self.SRmodels[-1].to(self.device)
                self.SRoptimizers.append(torch.optim.Adam(model.parameters(),lr=self.LR))
            self.patchinfo = np.load(self.patchinfo_dir)
            self.agent = agent.Agent(args,self.patchinfo.sum())

    #RANDOM MODEL INITIALIZATION FUNCTION
    def init_weights(self,m):
        if isinstance(m,torch.nn.Linear) or isinstance(m,torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight.data)

    #LOAD A CLEAN MODEL ACCORDING TO THE ARGUMENT
    def load(self,args):
        if args.model == 'ESRGAN':
            model = arch.RRDBNet(3,3,64,23,gc=32)
            model.load_state_dict(torch.load(SRMODEL_PATH),strict=True)
        elif args.model == 'random':
            model = arch.RRDBNet(3,3,64,23,gc=32)
            model.apply(self.init_weights)
        elif args.model == 'RCAN':
            torch.manual_seed(args.seed)
            checkpoint = utility.checkpoint(args)
            if checkpoint.ok:
                module = import_module('model.'+args.model.lower())
                model = module.make_model(args).to(self.device)
                kwargs = {}
                model.load_state_dict(torch.load(args.pre_train,**kwargs),strict=False)
            else: print('error loading RCAN model. QUITING'); quit();
        return model

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
    def train(self):
        #EACH EPISODE TAKE ONE LR/HR PAIR WITH CORRESPONDING PATCHES
        #AND ATTEMPT TO SUPER RESOLVE EACH PATCH

        #create our agent on based on previous information
        #requires pytorch 1.1.0+ which is not possible on the server
        #scheduler = torch.optim.lr_scheduler.CyclicLR(self.agent.optimizer,base_lr=0.0001,max_lr=0.1)
        #CREATE A TESTER to test at n iterations
        test = Tester(self.agent, self.SRmodels,testset=['Set5'])

        #START TRAINING
        for c in count():
            indices = list(range(len(self.TRAINING_HRPATH)))
            #FOR EACH HIGH RESOLUTION IMAGE
            for n,idx in enumerate(indices):
                HRpath = self.TRAINING_HRPATH[idx]
                LRpath = self.TRAINING_LRPATH[idx]
                LR = imageio.imread(LRpath)
                HR = imageio.imread(HRpath)
                LR,HR = self.getTrainingPatches(LR,HR)

                #WE MUST GO THROUGH EVERY SINGLE PATCH IN RANDOM ORDER WITHOUT REPLACEMENT
                patch_ids = list(range(len(LR)))
                random.shuffle(patch_ids)
                #S_loss = torch.zeros(1,requires_grad=True).float().to(self.device)
                while len(patch_ids) > 0:
                    sisr_loss = []
                    #batch_ids = patch_ids[-self.batch_size:]
                    #patch_ids = patch_ids[:-self.batch_size]
                    batch_ids = random.sample(patch_ids,self.batch_size)    #TRAIN ON A SINGLE IMAGE

                    labels = torch.Tensor(batch_ids).long().cuda()
                    lrbatch = LR[labels,:,:,:]
                    hrbatch = HR[labels,:,:,:]

                    #UPDATE OUR SISR MODELS
                    self.agent.opt.zero_grad()    #zero our policy gradients
                    for j,sisr in enumerate(self.SRmodels):
                        self.SRoptimizers[j].zero_grad()           #zero our sisr gradients
                        hr_pred = sisr(lrbatch)
                        m_labels = labels + int(np.sum(self.patchinfo[:idx]))

                        #update sisr model based on weighted l1 loss
                        l1diff = torch.abs(hr_pred - hrbatch).view(len(batch_ids),-1).mean(1)           #64x1 vector
                        onehot = torch.zeros(self.SR_COUNT); onehot[j] = 1.0                #1x4 vector
                        imgscore = torch.matmul(l1diff.unsqueeze(1),onehot.to(self.device).unsqueeze(0))    #64x4 matrix with column j as l1 diff and rest as zeros

                        weighted_imgscore = self.agent.model(imgscore,m_labels)     #do element wise matrix multiplication of l1 diff and softmax weights
                        loss1 = torch.mean(weighted_imgscore)
                        loss1.backward(retain_graph=True)
                        self.SRoptimizers[j].step()
                        sisr_loss.append(loss1.item())
                        #S_loss += loss1.data[0]

                    #gather the gradients of the agent policy and constrain them to be within 0-1 with max value as 1
                    one_matrix = torch.ones(len(batch_ids),self.SR_COUNT).to(self.device)
                    weight_identity = self.agent.model(one_matrix,m_labels)
                    val,maxid = weight_identity.max(1) #have max of each row equal to 1
                    loss3 = torch.mean(torch.abs(weight_identity[:,maxid] - 1))
                    loss3.backward()
                    self.agent.opt.step()

                    #LOG THE INFORMATION
                    agent_loss = loss3.item() + np.sum(sisr_loss)
                    print('\rEpoch/img: {}/{} | Agent Loss: {:.4f}, SISR Loss: {:.4f}'\
                          .format(c,n,agent_loss,np.mean(sisr_loss)),end="\n")

                    if self.logger:
                        self.logger.scalar_summary({'AgentLoss': torch.tensor([agent_loss]), 'SISRLoss': torch.tensor(np.mean(sisr_loss))})

                        #CAN'T QUITE GET THE ACTION VISUALIZATION WORK ON THE SERVER
                        #actions_taken = self.agent.model.M.weight.max(1)[1]
                        #self.logger.hist_summary('actions',np.array(actions_taken.tolist()),bins=self.SR_COUNT)
                        #self.logger.hist_summary('actions',actions_taken,bins=self.SR_COUNT)
                        self.logger.incstep()

                    #save the model after 200 images total of 800 images
                    if (self.logger.step+1) % 200 == 0:
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


