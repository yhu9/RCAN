#NATIVE IMPORTS
import os
import glob
from itertools import count
import random
import imageio

#OPEN SOURCE IMPORTS
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
    def getGroundTruthIOU(self,lr,hr,samplesize=0.1):
        if self.model == 'ESRGAN':
            lr = lr / 255.0
            hr = hr / 255.0
        #FORMAT THE INPUT
        lr, hr, info = util.getTrainingPatches(lr,hr,args,transform=False)

        # WE EVALUATE IOU ON ENTIRE IMAGE FED AS BATCHES OF PATCHES
        ids = list(range(len(lr)))
        batchsize = int(len(lr) * samplesize)
        patch_ids = list(range(len(hr)))
        score = 0
        for i in range(0,len(lr)-1,batchsize):
            batch_ids = torch.Tensor(patch_ids[i:i+batchsize]).long()

            LR = lr[batch_ids]
            HR = hr[batch_ids]
            LR = LR.to(self.device)
            HR = HR.to(self.device)

            #GET EACH SR RESULT
            choices = self.agent.model(LR)
            l1 = []
            for i, sisr in enumerate(self.SRmodels):
                l1.append(torch.abs(sisr(LR) - HR).mean(dim=1).mean(dim=1).mean(dim=1))
            l1diff = torch.stack(l1,dim=1)

            _,optimal_idx = l1diff.min(dim=1)
            _,predicted_idx = choices.max(dim=1)
            score += torch.sum((optimal_idx == predicted_idx).float()).item()
        IOU = score / len(lr)
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
        difficulty = sorted(difficulty,key=lambda x:x[1])
        return difficulty

    # OPTIMIZE SR MODELS AND SELECTION MODEL WITH CURRICULUM FOR 5 EPOCHS BY DEFAULT
    # input: data => 1D list of ints
    # output: none
    def optimize(self, data,epoch=5):
        # lossfn = torch.nn.L1Loss()

        for c in range(epoch):
            for n, idx in enumerate(data):
                hr_path = self.TRAINING_HRPATH[idx]
                lr_path = self.TRAINING_LRPATH[idx]
                lr = imageio.imread(lr_path)
                hr = imageio.imread(hr_path)

                lr, hr, _ = util.getTrainingPatches(lr,hr,args)

                patch_ids = list(range(len(lr)))
                random.shuffle(patch_ids)
                for step in range(1):
                    batch_ids = random.sample(patch_ids,self.batch_size)
                    labels = torch.Tensor(batch_ids).long()

                    lr_batch = lr[labels,:,:,:]
                    hr_batch = hr[labels,:,:,:]

                    lr_batch = lr_batch.to(self.device)
                    hr_batch = hr_batch.to(self.device)
                    if args.model == 'ESRGAN':
                        lr_batch = lr_batch / 255.0
                        hr_batch = hr_batch / 255.0

                    # GET SISR RESULTS FROM EACH MODEL
                    # loss_sisr = 0
                    sisrs = []

                    probs = self.agent.model(lr_batch)
                    for sisr in self.SRmodels:
                        hr_pred = sisr(lr_batch)
                        sisrs.append(hr_pred)

                    # UPDATE BOTH THE SISR MODELS AND THE SELECTION MODEL ACCORDING TO THEIR LOSS
                    sisr_loss = []
                    l1loss = []
                    # onehot_mask = torch.nn.functional.one_hot(maxarg,len(sisrs)).float()
                    for j, sr in enumerate(sisrs):
                        self.SRoptimizers[j].zero_grad()
                        l1 = torch.abs(sr - hr_batch).sum(dim=1).sum(dim=1).sum(dim=1) / ((self.PATCH_SIZE * self.UPSIZE)**2 * 3)
                        l1loss.append(l1)
                        loss = torch.mean(l1 * probs[:,j])
                        #loss = torch.mean(l1 * onehot_mask[:,j])
                        sisr_loss.append(loss)

                    l1loss = torch.stack(l1loss, dim=1)
                    self.agent.opt.zero_grad()
                    sisr_loss_total = sum(sisr_loss)
                    sisr_loss_total.backward()
                    [opt.step() for opt in self.SRoptimizers]
                    self.agent.opt.step()

                    #[sched.step() for sched in self.schedulers]
                    #self.agent.scheduler.step()

                    #CONSOLE OUTPUT FOR QUICK AND DIRTY DEBUGGING
                    lr = self.SRoptimizers[-1].param_groups[0]['lr']
                    lr2 = self.agent.opt.param_groups[0]['lr']
                    _, maxarg = probs[0].max(0)
                    sample_sr = sisrs[maxarg.item()][0]
                    sample_hr = hr_batch[0]
                    if args.model != 'ESRGAN':
                        sample_sr = sample_sr / 255.0
                        sample_hr = sample_hr / 255.0

                    choice = probs.max(dim=1)[1]
                    c1 = (choice == 0).float().mean()
                    c2 = (choice == 1).float().mean()
                    c3 = (choice == 2).float().mean()
                    s1 = torch.mean(l1loss[:,0]).item()
                    s2 = torch.mean(l1loss[:,1]).item()
                    s3 = torch.mean(l1loss[:,2]).item()
                    agentloss = torch.mean(l1loss.gather(1,choice.unsqueeze(1)))

                    print('\rEpoch/img: {}/{} | LR sr/ag: {:.8f}/{:.8f} | Agent Loss: {:.4f}, SISR Loss: {:.4f} | s1: {:.4f} | s2: {:.4f} | s3: {:.4f}'\
                            .format(c,n,lr,lr2,agentloss.item(),sisr_loss_total.item(),s1,s2,s3),end="\n")

                    #LOG AND SAVE THE INFORMATION
                    scalar_summaries = {'Loss/AgentLoss': agentloss, 'Loss/SISRLoss': sisr_loss_total, "choice/c1": c1, "choice/c2": c2, "choice/c3": c3, "sisr/s1": s1, "sisr/s2": s2, "sisr/s3": s3}
                    hist_summaries = {'actions': probs.view(-1), "choices": choice.view(-1)}
                    img_summaries = {'sr/HR': sample_hr.clamp(0,1),'sr/SR': sample_sr.clamp(0,1)}
                    self.logger.hist_summary(hist_summaries)
                    self.logger.scalar_summary(scalar_summaries)
                    self.logger.image_summary(img_summaries)
                    if self.logger.step % 100 == 0:
                        with torch.no_grad():
                            psnr,ssim,info = self.test.validateSet5(save=False,quick=False)
                        if self.logger:
                            self.logger.scalar_summary({'Testing_PSNR': psnr, 'Testing_SSIM': ssim})
                            weightedmask = torch.from_numpy(info['mask']).permute(2,0,1) / 255.0
                            mask = torch.from_numpy(info['maxchoice']).permute(2,0,1) / 255.0
                            optimal_mask = torch.from_numpy(info['optimalchoice']).permute(2,0,1) / 255.0
                            hrimg = torch.Tensor(info["HR"]).permute(2,0,1) / 255.0
                            srimg = torch.from_numpy(info['max']).permute(2,0,1) / 255.0
                            self.logger.image_summary({'Testing/Test Assignment':mask[:3],'Testing/Weight': weightedmask[:3], 'Testing/SR':srimg, 'Testing/HR': hrimg, 'Testing/optimalmask': optimal_mask})
                        self.savemodels()
                        self.agent.model.train()
                        [model.train() for model in self.SRmodels]

                    self.logger.incstep()

    # TRAINING REGIMEN
    def train(self,alpha=0, beta=3):
        # QUICK CHECK ON EVERYTHING
        #with torch.no_grad():
        #    psnr,ssim,info = self.test.validateSet5(save=False,quick=False)

        # START TRAINING
        data = set(range(len(self.TRAINING_HRPATH)))
        # lossfn = torch.nn.L1Loss()
        data.remove(alpha)
        curriculum = [alpha]
        self.optimize(curriculum,epoch=1000)

        # main training loop
        while True:

            # sorted training items in descending order of difficulty
            difficulty = self.getDifficulty(data)
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


