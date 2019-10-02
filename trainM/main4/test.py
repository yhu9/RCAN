#NATIVE IMPORTS
import os
import glob
import argparse

#OPEN SOURCE IMPORTS
import cv2
import numpy as np
import torch
import scipy.io as sio
import imageio
from importlib import import_module
import matplotlib.pyplot as plt
import matplotlib

#CUSTOM IMPORTS
import RRDBNet_arch as arch
from agent import Agent
from utils import util
from option import args
import utility

####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

#OUR GENERAL TESTING CLASS WHICH CAN BE USED DURING TRAINING AND EVALUTATION
class Tester():
    def __init__(self,agent=None,SRmodels=None,args=args,testset=['Set5','Set14','B100','Urban100','Manga109']):
        self.device = args.device
        if args.evaluate or args.viewM or args.testbasic:
            if args.model_dir == "": print("TRAINED AGENT AND SISR MODELS REQUIRED TO EVALUATE OR VIEW ALLOCATION"); quit();
            else:
                #LOADS THE TRAINED MODELS AND AGENT
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
        self.SRmodels = []
        loadedparams = torch.load(args.model_dir,map_location=self.device)
        self.patchinfo = np.load(args.patchinfo)
        self.agent = Agent(args,self.patchinfo.sum(),train=False,chkpoint=loadedparams)
        self.agent.model.eval()
        for i in range(args.action_space):
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
            model.load_state_dict(loadedparams["sisr" + str(i)])
            self.SRmodels.append(model)
            self.SRmodels[-1].to(self.device)
            self.SRmodels[-1].eval()

    #HELPER FUNCTION TO SHOW DECISION MAKING OF THE MODEL ON EVALUATION IMAGE
    def getPatchChoice(self,hr,info,save=False):
        cats = np.unique(info['assignment'])
        cat_colors = {0: [255,0,0], 1: [0,255,0], 2:[0,0,255], 3:[0,255,255], 4:[255,0,255],5:[255,255,0]}

        canvas = hr.copy().astype(np.uint8)
        mask = np.zeros(hr.shape).astype(np.uint8)
        gray = cv2.cvtColor(canvas,cv2.COLOR_RGB2GRAY).astype(np.uint8)
        h,w,d = hr.shape
        for c in cats:
            mask[info['assignment'] == c] = cat_colors[c]

        canvas = cv2.cvtColor(canvas,cv2.COLOR_BGR2HSV)
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2HSV)
        canvas[:,:,:2] = mask[:,:,:2]

        return cv2.cvtColor(canvas,cv2.COLOR_HSV2BGR)

    #EVALUATE A PARTICULAR MODEL USING FINAL G FUNCTION
    #NOT MADE YET
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
        info['SRimg'] = canvas.astype(np.uint8)
        return psnr,ssim,info

    #TEST A MODEL ON ALL DATASETS
    def validate(self,save=True,quick=False):
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
                hr = imageio.imread(hr_file)
                lr = imageio.imread(lr_file)
                psnr,ssim,info = self.evaluate_ideal(lr,hr)
                print(lr_file,psnr,ssim)
                scores[vset].append([psnr,ssim])

                #save info for each file tested
                if quick: break
                filename = os.path.join('runs',vset + os.path.basename(hr_file)+'.mat')
                if save:
                    info['LRimg'] = lr
                    info['HRimg'] = hr
                    info['LR_DIR'] = lr_file
                    info['HR_DIR'] = hr_file
                sio.savemat(filename,info)
                #cv2.imshow('Low Res',cv2.cvtColor(lr,cv2.COLOR_BGR2RGB))
                #cv2.imshow('High Res',cv2.cvtColor(hr,cv2.COLOR_BGR2RGB))
                #cv2.imshow('Super Res',cv2.cvtColor(info['SRimg'],cv2.COLOR_BGR2RGB))
                #cv2.waitKey(1)
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
    if args.evaluate:
        with torch.no_grad():
            testing_regime.validate()
    elif args.viewM:

        patchinfo = np.load(args.patchinfo)
        M = testing_regime.agent.getM()
        data = M.detach().cpu().numpy()
        data = data[:patchinfo[0],:]

        np.mean(data)
        np.std(data)
        matplotlib.use('tkagg')
        plt.hist(data,bins='auto')
        plt.title("histogram of the weight data")
        plt.show()

    elif args.testbasic:
        if args.hrimg != "":

            lrimg = imageio.imread(args.lrimg)
            hrimg = imageio.imread(args.hrimg)
            psnr,ssim,info = testing_regime.evaluate_ideal(lrimg,hrimg)
            choice = testing_regime.getPatchChoice(hrimg.astype(np.uint8),info)

            print("PSNR SCORE: {:.4f}".format(psnr) )
            print("SSIM SCORE: {:.4f}".format(ssim) )
            plt.figure(1)
            cv2.imshow('Choice Mask',cv2.cvtColor(choice,cv2.COLOR_BGR2RGB))
            cv2.imshow('Low Res',cv2.cvtColor(lrimg,cv2.COLOR_BGR2RGB))
            cv2.imshow('High Res',cv2.cvtColor(hrimg,cv2.COLOR_BGR2RGB))
            cv2.imshow('Super Res',cv2.cvtColor(info['SRimg'],cv2.COLOR_BGR2RGB))
            cv2.waitKey(0)
        elif args.lrimg != "":
            lrimg = imageio.imread(args.lrimg)
            print('your life is nothing')
        else:
            print('your life is nothing')





