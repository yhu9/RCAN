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
        SRMODEL_PATH = args.srmodel_path
        self.batchsize = args.batch_size
        self.TRAINING_LRPATH = glob.glob(os.path.join(args.training_lrpath,'x'+str(args.upsize),"*"))
        self.TRAINING_HRPATH = glob.glob(os.path.join(args.training_hrpath,'x'+str(args.upsize),"*"))
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
        self.k = args.action_space

        #DEFAULT START OR START ON PREVIOUSLY TRAINED EPOCH
        self.load(args)

        #INITIALIZE TESTING MODULE
        self.test = Tester(self.agent, self.SRmodels,args=args,testset=['Set5'])

    #LOAD A PRETRAINED AGENT WITH SUPER RESOLUTION MODELS
    def load(self,args):

        if args.model_dir != "":
            loadedparams = torch.load(args.model_dir,map_location=self.device)
            self.agent = agent.Agent(args,chkpoint=loadedparams)
            #self.agent = agent.Agent(args)
        else:
            self.agent = agent.Agent(args)
        self.SRmodels = []
        self.SRoptimizers = []
        self.schedulers = []
        for i in range(args.action_space):

            #CREATE THE ARCH
            if args.model == 'basic':
                model = arch.RRDBNet(3,3,32,args.d,gc=8)
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
                model.apply(init_weights)
            elif args.model == 'ESRGAN':
                model.load_state_dict(torch.load(args.ESRGAN_PATH),strict=True)
            elif args.model == 'RCAN':
                print('RCAN loaded!')
                model.load_state_dict(torch.load(args.pre_train,**kwargs),strict=True)
            elif args.model == 'basic':
                if args.d == 1:
                    model.load_state_dict(torch.load(args.basicpath_d1),strict=True)
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

            #self.SRoptimizers.append(torch.optim.Adam(model.parameters(),lr=1e-5))
            self.SRoptimizers.append(torch.optim.Adam(model.parameters(),lr=1e-5))
            scheduler = torch.optim.lr_scheduler.StepLR(self.SRoptimizers[-1],1000,gamma=0.5)

            self.schedulers.append(scheduler)

        #INCREMENT SCHEDULES TO THE CORRECT LOCATION
        #for i in range(args.step):
        #    [s.step() for s in self.schedulers]

    #TRAINING IMG LOADER WITH VARIABLE PATCH SIZES AND UPSCALE FACTOR
    def getTrainingPatches(self,LR,HR,transform=True):
        patch_size = self.PATCH_SIZE
        stride = self.PATCH_SIZE

        #RANDOMLY FLIP AND ROTATE IMAGE
        if transform:
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

    #EVALUATE THE INPUT AND GET SR RESULT
    def getGroundTruthIOU(self,lr,hr,samplesize=10,it=1):
        if self.model == 'ESRGAN' or self.model == 'basic':
            lr = lr / 255.0
            hr = hr / 255.0

        #FORMAT THE INPUT
        lr, hr = self.getTrainingPatches(lr,hr)

        # WE EVALUATE IOU ON ENTIRE IMAGE FED AS BATCH OF PATCHES
        # batchsize = int(len(lr) * samplesize)
        batchsize = samplesize
        score = 0
        for i in range(1):
            patch_ids = random.sample(list(range(len(hr))), batchsize)
            batch_ids = torch.Tensor(patch_ids).long()

            LR = lr[batch_ids]
            HR = hr[batch_ids]
            LR = LR.to(self.device)
            HR = HR.to(self.device)

            #GET EACH SR RESULT
            sisrs = []
            l1 = []
            for i, sisr in enumerate(self.SRmodels):
                sr_result = sisr(LR)
                sisrs.append(sr_result)
                l1.append(torch.abs(sr_result - HR).mean(dim=1))
            sisrs = torch.cat(sisrs,dim=1)
            choices = self.agent.model(sisrs)
            l1diff = torch.stack(l1,dim=1)

            _,optimal_idx = l1diff.min(dim=1)
            _,predicted_idx = choices.max(dim=1)
            score += torch.sum((optimal_idx == predicted_idx).float()).item()
        h,w = optimal_idx.shape[-2:]
        IOU = score / (batchsize * h*w)
        return IOU

    # GATHER THE DIFFICULTY OF THE DATASET
    def getDifficulty(self,DATA):
        [sisr.eval() for sisr in self.SRmodels]
        self.agent.model.eval()
        difficulty = []
        for d in DATA:

            lrpath = self.TRAINING_LRPATH[d]
            hrpath = self.TRAINING_HRPATH[d]
            LR = imageio.imread(lrpath)
            HR = imageio.imread(hrpath)

            with torch.no_grad():
                iou = self.getGroundTruthIOU(LR,HR)

            difficulty.append( (d,iou) )
            print(f"Image: {d}, IOU: {iou:.4f}")
        difficulty = sorted(difficulty,key=lambda x:x[1])
        [sisr.train() for sisr in self.SRmodels]
        self.agent.model.train()
        return difficulty

    # objective defined by IIC paper "Invariant Information Clustering for Unsupervised IMage Classification and Segmentation"
    def mutualInformation(self,z,zt):
        P = (z.unsqueeze(2) * zt.unsqueeze(1)).sum(dim=0)
        P = P / P.sum()
        P = P.clamp(min=1e-16)
        Pi = P.sum(dim=1).view(self.k,1).expand(self.k,2)
        Pj = P.sum(dim=0).view(1,2).expand(self.k,2)
        return -1 * (P * (torch.log(P) - (torch.log(Pi) + torch.log(Pj)))).sum()

    # conditional entropy
    def conditionalEntropy(self,z,zt):
        P = (z.unsqueeze(2) * zt.unsqueeze(1)).sum(dim=0)
        P = P / P.sum()
        P = P.clamp(min=1e-16)
        Pk = P.sum(dim=1).view(self.k,1).expand(self.k,2)
        return -1 * (P * (torch.log(P) - torch.log(Pk))).sum()

    # get probabitilies based on trained agent
    def getProbs(self,lrbatch):
        prob = self.agent.model(lrbatch)
        mode = int(random.random()*10) % 3
        if mode == 0:
            prob2 = util.rotateTensor(self.agent.model(util.rotateTensor(lrbatch,90)),90)
        elif mode == 1:
            prob2 = util.rotateTensor(self.agent.model(util.rotateTensor(lrbatch,180)),180)
        elif mode == 2:
            prob2 = util.rotateTensor(self.agent.model(util.rotateTensor(lrbatch,270)),270)
        return prob,prob2

    # get data
    def getBatch(self,data):
        # GET INPUT FROM CURRENT IMAGE
        idx = random.sample(data,1)[0]
        HRpath = self.TRAINING_HRPATH[idx]
        LRpath = self.TRAINING_LRPATH[idx]
        LR = imageio.imread(LRpath)
        HR = imageio.imread(HRpath)
        if self.model == 'ESRGAN' or self.model == 'basic':
            LR = LR / 255.0
            HR = HR / 255.0
        LR,HR = self.getTrainingPatches(LR,HR)
        patch_ids = list(range(len(LR)))

        batch_ids = random.sample(patch_ids,self.batchsize)
        labels = torch.Tensor(batch_ids).long()

        lrbatch = LR[labels,:,:,:]
        hrbatch = HR[labels,:,:,:]
        lrbatch = lrbatch.to(self.device)
        hrbatch = hrbatch.to(self.device)

        return lrbatch,hrbatch

    #TRAINING REGIMEN
    def optimize(self,data,iou_threshold=0.7,miniter=500):
        #EACH EPISODE TAKE ONE LR/HR PAIR WITH CORRESPONDING PATCHES
        #AND ATTEMPT TO SUPER RESOLVE EACH PATCH

        #QUICK CHECK ON EVERYTHING
        with torch.no_grad():
            psnr,ssim,info = self.test.validateSet5(save=False,quick=False)

        agent_iou = deque(maxlen=miniter)
        while True:

            # get data
            lrbatch,hrbatch = self.getBatch(data)

            # gather probabilities
            probs = self.agent.model(lrbatch)

            # GET SISR RESULTS FROM EACH MODEL
            sisrs = []
            for j, sisr in enumerate(self.SRmodels):
                hr_pred = sisr(lrbatch)
                sisrs.append(hr_pred)
            sr_results = torch.cat(sisrs,dim=1)
            choice_val, choice = probs.max(dim=1)

            # UPDATE BOTH THE SISR MODELS AND THE SELECTION MODEL ACCORDING TO THEIR LOSS
            SR_result = torch.zeros(self.batchsize,3,self.PATCH_SIZE * self.UPSIZE,self.PATCH_SIZE * self.UPSIZE).to(self.device)
            l2diff = []
            for j, sr in enumerate(sisrs):
                self.SRoptimizers[j].zero_grad()
                diff = (sr-hrbatch).pow(2).sum(dim=1)
                l2diff.append(diff)
                pred = sr * 1/len(sisrs)
                SR_result += pred
            l2diff = torch.stack(l2diff, dim=1)
            diffmap = torch.nn.functional.softmax(-1 * (l2diff - torch.mean(l2diff)),dim=1)
            self.agent.opt.zero_grad()
            minval, minidx = l2diff.min(dim=1)

            # cross over
            scores = l2diff.view(self.batchsize,-1).mean(dim=-1)
            _,best_idx = torch.max(scores,0)
            param1 = self.SRmodels[best_idx].named_parameters()
            for i,sisr in enumerate(self.SRmodels):
                if i == best_idx: continue
                param2 = sisr.named_parameters()
                dict_param2 = dict(param2)
                for name,w in param1:
                    dict_param2[name].data.copy_((w.data + dict_param2[name].data)*0.5)
                sisr.load_state_dict(dict_param2)

            # another option is to minimize intraclass/interclass probabilities
            #sisrloss = torch.mean(minval)
            #sisrloss = torch.sum(minval)
            #sisrloss = torch.mean(l2diff.gather(1,choice.unsqueeze(1)))                                                 # prob max sum
            #sisrloss = torch.mean(l2diff.gather(1,choice.unsqueeze(1)))                                                 # prob max
            #sisrloss = torch.nn.functional.l1_loss(SR_result,hrbatch)                                                           # ensemble
            #sisrloss = torch.sum(l2diff.gather(1,choice.unsqueeze(1))) + torch.sum(minval)
            sisrloss = torch.mean(l2diff.gather(1, choice.unsqueeze(1))) + 1000*torch.nn.functional.l1_loss(SR_result,hrbatch)       # mean loss    *original
            #sisrloss = torch.mean(l2diff.gather(1, choice.unsqueeze(1))) + torch.nn.functional.l1_loss(SR_result,hrbatch)               # mean loss no alpha

            selectionloss = torch.mean(-1 * (probs.gather(1,minidx.unsqueeze(1)) + 1e-16).log())        # cross entropy loss with one hot ground truth
            #selectionloss = self.conditionalEntropy(probs.reshape(-1,self.k),correct)
            #selectionloss = cond_info + torch.mean(-1 * (probs.gather(1,minidx.unsqueeze(1)) + 1e-16).log())
            #selectionloss = self.mutualInformation(probs.reshape(-1,self.k),correct)

            #sisrloss_total = sisrloss
            sisrloss_total = sisrloss + selectionloss
            sisrloss_total.backward()
            [opt.step() for opt in self.SRoptimizers]
            self.agent.opt.step()


            target = torch.zeros(l2diff.shape)
            for i in range(len(self.SRmodels)):
                target[:,i] = (minidx == i).float()
            selectionloss = selectionloss.mean()

            # CONSOLE OUTPUT FOR QUICK AND DIRTY DEBUGGING
            lr = self.SRoptimizers[-1].param_groups[0]['lr']
            lr2 = self.agent.opt.param_groups[0]['lr']
            if not(self.model == 'ESRGAN' or self.model == 'basic'):
                SR_result = SR_result / 255
                hrbatch = hrbatch / 255
            iou = (choice == minidx).float().sum() / (choice.shape[0] * choice.shape[1] * choice.shape[2])
            agent_iou.append(iou.item())
            c1 = (choice == 0).float().mean()
            c2 = (choice == 1).float().mean()
            c3 = (choice == 2).float().mean()
            print('\rdata size/img: {}/{} | LR sr/ag: {:.8f}/{:.8f} | Agent Loss: {:.4f}, SISR Loss: {:.4f},  Total: {:.4f}| IOU: {:.4f} | c1: {:.4f}, c2: {:.4f}, c3: {:.4f}'\
                    .format(len(data),self.logger.step,lr,lr2,selectionloss.item(),sisrloss.item(), sisrloss_total.item(), np.mean(agent_iou), c1.item(), c2.item(), c3.item()),end="\n")

            # LOG AND SAVE THE INFORMATION
            scalar_summaries = {'Loss/AgentLoss': selectionloss, 'Loss/sisrloss_total': sisrloss_total,'Loss/sisrloss': sisrloss, 'Loss/IOU': iou, "choice/c1": c1, "choice/c2": c2, "choice/c3": c3}
            #hist_summaries = {'actions': probs[0].view(-1), "choices": choice[0].view(-1)}
            img_summaries = {'sr/mask': probs[0][:3], 'sr/sr': SR_result[0][:3].clamp(0,1),'sr/targetmask': target[0][:3], 'sr/diffmap': diffmap[0][:3]}
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
                    srimg = torch.from_numpy(info['weighted'] ).permute(2,0,1).clamp(0,1)
                    self.logger.image_summary({'Testing/Test Assignment':mask[:3], 'Testing/SR':srimg, 'Testing/HR': hrimg, 'Testing/upperboundmask': best_mask[:3]})
                self.savemodels()
            self.logger.incstep()

            if np.mean(agent_iou) >= iou_threshold and len(agent_iou) == miniter: break

    # TRAINING REGIMEN
    def train(self,alpha=0, beta=20):
        data = set(range(len(self.TRAINING_HRPATH)))
        data.remove(alpha)
        #curriculum = [alpha]
        curriculum = list(range(len(self.TRAINING_HRPATH)))
        self.optimize(curriculum,1.0)
        # main training loop
        for i in count():
            difficulty = self.getDifficulty(data)
            print("ADDED NEXT EASIEST")
            np.save('runs/curriculum_' + str(i),np.array(difficulty))
            A = [a[0] for a in difficulty[-beta:]]
            curriculum += A
            [data.remove(a) for a in A]
            self.optimize(curriculum,0.4)

########################################################################################################
########################################################################################################
########################################################################################################
if __name__ == '__main__':
    sisr = SISR()
    sisr.train()


