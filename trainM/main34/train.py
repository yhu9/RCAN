#NATIVE IMPORTS
import os
import glob
from itertools import count
import random
import imageio
from collections import deque

#OPEN SOURCE IMPORTS
import numpy as np
import torch

#CUSTOM IMPORTS
import RRDBNet_arch as arch
import agent
import logger
import utility
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
            #self.agent = agent.Agent(args,chkpoint=loadedparams)
            self.agent = agent.Agent(args)
        else:
            self.agent = agent.Agent(args)
        self.SRmodels = []
        self.SRoptimizers = []
        self.schedulers = []
        for i in range(args.action_space):

            #CREATE THE ARCH
            if args.model == 'ESRGAN':
                model = arch.RRDBNet(3,3,64,23,gc=32)
            if args.model == 'basic':
                model = arch.RRDBNet(3,3,32,args.d,gc=8,upsize=args.upsize)
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
                print('loading basic model')
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

            self.SRoptimizers.append(torch.optim.Adam(model.parameters(),lr=1e-5))
            scheduler = torch.optim.lr_scheduler.StepLR(self.SRoptimizers[-1],200,gamma=0.8)

            self.schedulers.append(scheduler)

        #INCREMENT SCHEDULES TO THE CORRECT LOCATION
        for i in range(args.step):
            [s.step() for s in self.schedulers]

    # SAVE THE AGENT AND THE SISR MODELS INTO A SINGLE FILE
    def savemodels(self):
        data = {'agent': self.agent.model.state_dict()}
        for i,m in enumerate(self.SRmodels):
            modelname = "sisr" + str(i)
            data[modelname] = m.state_dict()
        data['step'] = self.logger.step
        torch.save(data,"models/" + self.name + "_sisr.pth")

    # CREATE A CURICULUM ON ADDITIONAL DATA
    # def curriculum(self,addtional_data):
    def getTrainingIndices(self):
        indices = list(range(len(self.TRAINING_HRPATH)))
        data = []
        for idx in indices:
            HRpath = self.TRAINING_HRPATH[idx]
            LRpath = self.TRAINING_LRPATH[idx]
            LR = imageio.imread(LRpath)
            HR = imageio.imread(HRpath)
            LR,HR,_ = util.getTrainingPatches(LR,HR,args)

            data.append(range(len(LR)))
        return data

    #EVALUATE THE INPUT AND GET SR RESULT
    def getGroundTruthIOU(self,lr,hr,samplesize=0.01):
        if self.model == 'ESRGAN' or self.model == 'basic':
            lr = lr / 255.0
            hr = hr / 255.0

        #FORMAT THE INPUT
        lr, hr, info = util.getTrainingPatches(lr,hr,args,transform=False)

        # WE EVALUATE IOU ON ENTIRE IMAGE FED AS BATCH OF PATCHES
        #batchsize = int(len(lr) * samplesize)
        maxsize = int(len(lr) * samplesize)
        patch_ids = list(range(len(hr)))
        score = 0
        batch_ids = torch.Tensor(random.sample(patch_ids,maxsize)).long()

        #for i in range(0,maxsize-1,self.batch_size):
        #batch_ids = torch.Tensor(patch_ids[i:i+self.batch_size]).long()

        LR = lr[batch_ids]
        HR = hr[batch_ids]
        LR = LR.to(self.device)
        HR = HR.to(self.device)

        #GET EACH SR RESULT
        choices = self.agent.model(LR)
        diff = []
        for i, sisr in enumerate(self.SRmodels):
            error = (sisr(LR) - HR).pow(2).mean(dim=1).mean(dim=1).mean(dim=1)      # MSE
            #error = (torch.abs(sisr(LR) - HR).mean(dim=1).mean(dim=1).mean(dim=1)  # MAE
            diff.append(error)
        l1diff = torch.stack(diff,dim=1)

        _,optimal_idx = l1diff.min(dim=1)
        _,predicted_idx = choices.max(dim=1)
        score += torch.sum((optimal_idx == predicted_idx).float()).item()

        # output IOU
        IOU = score / maxsize
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
            print(d,iou)
        difficulty = sorted(difficulty,key=lambda x:x[1])
        return difficulty

    #STANDARDIZE TRAINING DATA WITH MEAN AND STD
    def standardize(self,x):
        mean = []
        std = []

    # OPTIMIZE SR MODELS AND SELECTION MODEL WITH CURRICULUM FOR 5 EPOCHS BY DEFAULT
    # input: data => 1D list of ints
    # output: none
    def optimize(self, data,iou_threshold=0.5):
        self.agent.model.train()
        [model.train() for model in self.SRmodels]
        agent_iou = deque(maxlen=100)

        # while the agent iou is not good enough
        for c in count():
            # get an image
            idx = random.sample(data,1)[0]

            hr_path = self.TRAINING_HRPATH[idx]
            lr_path = self.TRAINING_LRPATH[idx]
            lr = imageio.imread(lr_path)
            hr = imageio.imread(hr_path)

            lr, hr, _ = util.getTrainingPatches(lr,hr,args,transform=False)

            patch_ids = list(range(len(lr)))
            random.shuffle(patch_ids)

            # get the mini batch
            batch_ids = random.sample(patch_ids,self.batch_size)
            labels = torch.Tensor(batch_ids).long()
            lr_batch = lr[labels,:,:,:]
            hr_batch = hr[labels,:,:,:]
            lr_batch = lr_batch.to(self.device)
            hr_batch = hr_batch.to(self.device)
            if args.model == 'ESRGAN' or args.model == 'basic':
                lr_batch = lr_batch / 255.0
                hr_batch = hr_batch / 255.0

            # UPDATE THE SISR MODELS
            self.agent.opt.zero_grad()
            sr_result = torch.zeros(self.batch_size,3,self.PATCH_SIZE * self.UPSIZE,self.PATCH_SIZE * self.UPSIZE).to(self.device)
            sr_result.requires_gard = False
            probs = self.agent.model(lr_batch)
            #sisr_loss = 0
            sisrs = []
            pred_diff = []
            for j,sisr in enumerate(self.SRmodels):
                self.SRoptimizers[j].zero_grad()
                hr_pred = sisr(lr_batch)
                diff = (hr_pred - hr_batch).pow(2).mean(dim=1).mean(dim=1).mean(dim=1)    #MEAN OF FROB NORM SQUARED ACROSS CxHxW
                #diff = torch.abs(hr_pred - hr_batch).sum(dim=1).sum(dim=1).sum(dim=1) / ((self.PATCH_SIZE * self.UPSIZE)**2 * 3)   #MAE ACROSS CxHxW
                #sisr_loss += torch.mean(diff * probs[:,j])
                pred_diff.append(diff)
                sisrs.append(hr_pred)
            pred_diff = torch.stack(pred_diff, dim=1)
            minval, optimalidx = pred_diff.min(dim=1)
            selectionloss = torch.mean(probs.gather(1,optimalidx.unsqueeze(1)).clamp(1e-16,1).log()) * -1
            sisrloss = minval.mean()
            sisr_loss_total = sisrloss + selectionloss
            sisr_loss_total.backward()
            [opt.step() for opt in self.SRoptimizers]
            self.agent.opt.step()

            # VISUALIZATION
            # CONSOLE OUTPUT FOR QUICK AND DIRTY DEBUGGING
            lr1 = self.SRoptimizers[-1].param_groups[0]['lr']
            lr2 = self.agent.opt.param_groups[0]['lr']
            _, maxarg = probs[0].max(0)
            sample_sr = sisrs[maxarg.item()][0]
            sample_hr = hr_batch[0]
            if args.model != 'ESRGAN' and args.model != 'basic':
                sample_sr = sample_sr / 255.0
                sample_hr = sample_hr / 255.0

            choice = probs.max(dim=1)[1]
            iou = (choice == optimalidx).float().sum() / (len(choice))
            c1 = (choice == 0).float().mean()
            c2 = (choice == 1).float().mean()
            c3 = (choice == 2).float().mean()
            s1 = torch.mean(pred_diff[:,0]).item()
            s2 = torch.mean(pred_diff[:,1]).item()
            s3 = torch.mean(pred_diff[:,2]).item()

            agent_iou.append(iou.item())

            print('\rEpoch/img: {}/{} | LR: {:.8f} | Agent Loss: {:.4f}, SISR Loss: {:.4f} | IOU: {:.4f} | c1: {:.4f}, c2: {:.4f}, c3: {:.4f}'\
                        .format(c,self.logger.step,lr2,selectionloss.item(),sisrloss.item(), np.mean(agent_iou), c1.item(), c2.item(), c3.item()),end="\n")
            #print('\rEpoch/img: {}/{} | LR sr/ag: {:.8f}/{:.8f} | Agent Loss: {:.4f} | SISR Loss: {:.4f} | IOU : {:.4f} | s1: {:.4f} | s2: {:.4f} | s3: {:.4f}'\
            #        .format(c,self.logger.step,lr1,lr2,sisr_loss_total.item(),sisr_loss_total.item(),np.mean(agent_iou),s1,s2,s3),end="\n")

            #LOG AND SAVE THE INFORMATION
            scalar_summaries = {'Loss/AgentLoss': sisr_loss_total, 'Loss/SISRLoss': sisr_loss_total,"Loss/IOU": np.mean(agent_iou), "choice/c1": c1, "choice/c2": c2, "choice/c3": c3, "sisr/s1": s1, "sisr/s2": s2, "sisr/s3": s3}
            #hist_summaries = {'actions': probs.view(-1), "choices": choice.view(-1)}
            img_summaries = {'sr/HR': sample_hr.clamp(0,1),'sr/SR': sample_sr.clamp(0,1)}
            #self.logger.hist_summary(hist_summaries)
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
                    optimal_mask = torch.from_numpy(info['optimalchoice']).permute(2,0,1) /255.0
                    hrimg = torch.Tensor(info["HR"]).permute(2,0,1)
                    srimg = torch.from_numpy(info['max']).permute(2,0,1) / 255.0
                    self.logger.image_summary({'Testing/Test Assignment':mask[:3], 'Testing/Weight': weightedmask[:3], 'Testing/SR':srimg, 'Testing/HR': hrimg, 'Testing/optimalmask': optimal_mask[:3]})
                    #self.logger.image_summary({'Testing/Test Assignment':mask[:3], 'Testing/SR':srimg, 'Testing/HR': hrimg, 'Testing/upperboundmask': best_mask[:3]})
                self.savemodels()
            self.logger.incstep()

            if np.mean(agent_iou) > iou_threshold or c+1 % 10000 == 0: break

    # TRAINING REGIMEN
    def train(self,alpha=0, beta=10):
        # QUICK CHECK ON EVERYTHING
        #with torch.no_grad():
        #    psnr,ssim,info = self.test.validateSet5(save=False,quick=False)

        # START TRAINING
        #data = set(range(len(self.TRAINING_HRPATH))[:10])
        data = set(range(len(self.TRAINING_HRPATH)))
        #data.remove(alpha)
        curriculum = [alpha]
        self.optimize(curriculum,iou_threshold=0.5)

        # main training loop
        while True:
            # sorted training items in descending order of difficulty
            difficulty = self.getDifficulty(data)
            A = [a[0] for a in difficulty[-beta:]]
            curriculum += A
            [data.remove(a) for a in A]
            self.optimize(curriculum,iou_threshold=0.5)

########################################################################################################
########################################################################################################
########################################################################################################
if __name__ == '__main__':

    sisr = SISR()
    sisr.train()


