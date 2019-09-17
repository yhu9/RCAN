#NATIVE IMPORTS
import os
import glob
import argparse

#OPEN SOURCE IMPORTS
import cv2
import numpy as np
import torch
import scipy.io as sio

#CUSTOM IMPORTS
import RRDBNet_arch as arch
from agent import Agent
from utils import util
from option import args

####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

#OUR GENERAL TESTING CLASS WHICH CAN BE USED DURING TRAINING AND EVALUTATION
class Tester():
    def __init__(self,agent=None,SRmodels=None,args=args,evaluate=True,testset=['Set5','Set14','B100','Urban100','Manga109']):
        self.device = args.device
        if evaluate:
            if args.model_dir == "": print("TRAINED AGENT AND SISR MODELS REQUIRED TO EVALUATE"); quit();
            self.load(args)
        else:
            if not agent or not SRmodels: print("TRAINED AGENT AND SISR MODELS REQUIRED"); quit();
            self.agent = agent
            self.SRmodels = SRmodels

        downsample_method = args.down_method
        self.hr_rootdir = os.path.join(args.dataroot,'HR')
        self.lr_rootdir = os.path.join(args.dataroot,"LR" + downsample_method)
        self.validationsets = testset
        self.upsize = args.upsize
        self.resfolder = 'x' + str(args.upsize)
        self.PATCH_SIZE = args.patchsize

    #LOAD A PRETRAINED AGENT WITH SUPER RESOLUTION MODELS
    def load(self,args):
        loadedparams = torch.load(args.model_dir,map_location=self.device)
        self.patchinfo = np.load(args.patchinfo)
        self.agent = Agent(args,self.patchinfo.sum(),train=False)
        self.agent.model.eval()
        self.SRmodels = []
        for i in range(args.action_space):
            model = arch.RRDBNet(3,3,64,23,gc=32)
            model.load_state_dict(loadedparams["sisr" + str(i)])
            self.SRmodels.append(model)
            self.SRmodels[-1].to(self.device)
            self.SRmodels[-1].eval()

    #EVALUATE A PARTICULAR MODEL USING FINAL G FUNCTION
    def evaluate(self,model,lr,hr):
        h,w,d = lr.shape

    #EVALUATE THE IDEAL CASE WHERE WE ACCURATELY IDENTIFY THE IDEAL MODEL K FOR PATCH N
    def evaluate_ideal(self,lr,hr):
        lr = lr * 1.0 / 255
        h,w,d = lr.shape
        hrh,hrw,_ = hr.shape
        canvas = np.zeros((h*4,w*4,3))
        size = self.PATCH_SIZE
        stride = self.PATCH_SIZE // 2
        pad = self.PATCH_SIZE // 8
        info = {'assignment': np.zeros((hrh,hrw))}

        for i in range(0,h-1,stride):
            for j in range(0,w-1,stride):

                lr_img = lr[i:i+size] if i+size < h else lr[i:]
                lr_img = lr_img[:,j:j+size,:] if j+size < w else lr_img[:,j:]
                lr_img = torch.from_numpy(np.transpose(lr_img[:,:,[2,1,0]],(2,0,1))).float()
                lr_img = lr_img.unsqueeze(0)
                lr_img = lr_img.to(self.device)
                hr_img = hr[i*self.upsize:(i+size)*self.upsize,j*self.upsize:(j+size)*self.upsize,:]

                psnrscores = []
                ssimscores = []
                sr_predictions = []
                for sisr in self.SRmodels:
                    hr_hat = sisr(lr_img).data.squeeze().float().cpu().clamp_(0,1).numpy()
                    hr_hat = np.transpose(hr_hat[[2,1,0],:,:],(1,2,0))
                    hr_hat = (hr_hat * 255.0).round()
                    sr_predictions.append(hr_hat)

                    psnr,ssim = util.calc_metrics(hr_img,hr_hat,crop_border=self.upsize)
                    psnrscores.append(psnr)
                    ssimscores.append(ssim)

                top,top_= (0,0) if i == 0 else ((i+pad)*self.upsize,pad*self.upsize)
                bot,bot_ = (hrh,size*self.upsize) if i+size >= h else ((i+size-pad)*self.upsize,-pad*self.upsize)
                lef,lef_ = (0,0) if j == 0 else ((j+pad)*self.upsize,pad*self.upsize)
                rig,rig_ = (hrw,size*self.upsize) if j+size >= w else ((j+size-pad)*self.upsize,-pad*self.upsize)

                idx = psnrscores.index(max(psnrscores))
                canvas[top:bot,lef:rig] = sr_predictions[idx][top_:bot_,lef_:rig_]
                info['assignment'][top:bot,lef:rig] = idx

                #cv2.imshow('srimg',canvas.astype(np.uint8))
                #cv2.imshow('gtimg',hr.astype(np.uint8))
                #cv2.waitKey(1)

        psnr,ssim = util.calc_metrics(hr,canvas,crop_border=self.upsize)
        info['psnr'] = psnr
        info['ssim'] = ssim
        print(psnr,ssim)
        return psnr,ssim,info

    #TEST A MODEL ON ALL DATASETS
    def validate(self,save=True):
        scores = {}
        [model.eval() for model in self.SRmodels]
        for vset in self.validationsets:
            scores[vset] = []
            HR_dir = os.path.join(self.hr_rootdir,vset)
            LR_dir = os.path.join(os.path.join(self.lr_rootdir,vset),self.resfolder)

            #SORT THE HR AND LR FILES IN THE SAME ORDER
            HR_files = [os.path.join(HR_dir, f) for f in os.listdir(HR_dir)]
            LR_files = [os.path.join(LR_dir, f) for f in os.listdir(LR_dir)]
            HR_files.sort()
            LR_files.sort()

            #APPLY SISR ON EACH LR IMAGE AND GATHER RESULTS
            for hr_file,lr_file in zip(HR_files,LR_files):
                hr = cv2.imread(hr_file,cv2.IMREAD_COLOR)
                lr = cv2.imread(lr_file,cv2.IMREAD_COLOR)
                psnr,ssim,info = self.evaluate_ideal(lr,hr)
                scores[vset].append([psnr,ssim])

                #save distribution info of psnr scores for each file tested
                filename = os.path.join('runs',vset + os.path.basename(hr_file)+'.mat')
                if save:
                    info['LR_DIR'] = lr_file
                    info['HR_DIR'] = hr_file
                sio.savemat(filename,info)

            mu_psnr = np.mean(np.array(scores[vset])[:,0])
            mu_ssim = np.mean(np.array(scores[vset])[:,1])
            print(vset + ' scores',mu_psnr,mu_ssim)
        return mu_psnr,mu_ssim

####################################################################################################
####################################################################################################
####################################################################################################

if __name__ == '__main__':
    ####################################################################################################
    ####################################################################################################

    testing_regime = Tester()
    with torch.no_grad():
        testing_regime.validate()




