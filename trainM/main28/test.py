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
#testset=['Set5','Set14','B100','Urban100','Manga109']
class Tester():
    def __init__(self,agent=None,SRmodels=None,args=args,testset=['Set5','Set14','B100','Urban100', 'Manga109']):
        self.device = args.device
        if args.evaluate or args.ensemble or args.viewM or args.testbasic or args.baseline:
            if args.model_dir == "" and not args.baseline: print("TRAINED AGENT AND SISR MODELS REQUIRED TO EVALUATE OR VIEW ALLOCATION"); quit();
            else:
                #LOADS THE TRAINED MODELS AND AGENT
                self.load(args)
        else:
            if not agent or not SRmodels: print("TRAINED AGENT AND SISR MODELS REQUIRED"); quit();
            self.agent = agent
            self.SRmodels = SRmodels

        downsample_method = args.down_method
        self.hr_rootdir = os.path.join(args.dataroot,'HR2')
        self.lr_rootdir = os.path.join(args.dataroot,"LR" + downsample_method)
        self.validationsets = testset
        self.upsize = args.upsize
        self.resfolder = 'x' + str(args.upsize)
        self.PATCH_SIZE = args.patchsize
        self.name = args.name
        self.model = args.model

    #LOAD A PRETRAINED AGENT WITH SUPER RESOLUTION MODELS
    def load(self,args):
        self.SRmodels = []

        if not args.baseline:
            loadedparams = torch.load(args.model_dir,map_location=self.device)
            self.agent = Agent(args,chkpoint=loadedparams)
            self.agent.model.eval()

        for i in range(args.action_space):
            if args.model == 'ESRGAN':
                model = arch.RRDBNet(3,3,64,23,gc=32,upsize=args.upsize)   #
            elif args.model == 'basic':
                model = arch.RRDBNet(3,3,32,args.d,gc=8,upsize=args.upsize)
            elif args.model == 'RCAN':
                torch.manual_seed(args.seed)
                checkpoint = utility.checkpoint(args)
                if checkpoint.ok:
                    module = import_module('model.'+args.model.lower())
                    model = module.make_model(args).to(self.device)
                    kwargs = {}
                else: print('error loading RCAN model. QUITING'); quit();

            if args.baseline and args.model == 'ESRGAN':
                model.load_state_dict(torch.load(args.ESRGAN_PATH),strict=True)
            elif args.baseline and args.model == 'RCAN':
                print('rcan loaded')
                model.load_state_dict(torch.load(args.pre_train),strict=False)
            elif args.model == 'bicubic':
                return
            else:
                model.load_state_dict(loadedparams["sisr" + str(i)])

            self.SRmodels.append(model)
            self.SRmodels[-1].to(self.device)
            self.SRmodels[-1].eval()

    #HELPER FUNCTION TO SHOW DECISION MAKING OF THE MODEL ON EVALUATION IMAGE
    def getPatchChoice(self,info,save=False):
        canvas = info['SRimg'].copy().astype(np.uint8)
        gray = cv2.cvtColor(canvas,cv2.COLOR_RGB2GRAY).astype(np.uint8)
        mask = (info['assignment'] * 255.0).round().astype(np.uint8)

        #GET CHANNELS
        gray = cv2.cvtColor(canvas,cv2.COLOR_RGB2GRAY).astype(np.uint8)
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2HSV).astype(np.uint8)

        #PAINT OUR CANVAS
        canvas = cv2.cvtColor(canvas,cv2.COLOR_BGR2HSV)
        canvas[:,:,:2] = mask[:,:,:2]
        return cv2.cvtColor(canvas,cv2.COLOR_HSV2BGR)

    #HELPER FUNCTION TO GET LOCAL SCORE OF THE IMAGE
    def getLocalScore(self,SR,HR,size=16,step=8):
        bound = step
        srimg = util.rgb2ycbcr(SR)
        hrimg = util.rgb2ycbcr(HR)

        #Sliding window approach to find local psnr and ssim values
        srimg = np.pad(srimg,pad_width=((bound,bound),(bound,bound)),mode='symmetric')
        hrimg = np.pad(hrimg,pad_width=((bound,bound),(bound,bound)),mode='symmetric')
        h,w = hrimg.shape[:2]
        psnr_vals = np.zeros(hrimg.shape)
        ssim_vals = np.zeros(hrimg.shape)
        for i in range(bound,h-bound-1,1):
            for j in range(bound,w-bound-1,1):
                img1 = srimg[i-bound:i+bound+1,j-bound:j+bound+1]
                img2 = hrimg[i-bound:i+bound+1,j-bound:j+bound+1]
                psnr_vals[i,j] = util.calc_psnr(img1 * 255,img2 * 255)
                ssim_vals[i,j] = util.calc_ssim(img1 * 255,img2 * 255)
        psnr_vals = psnr_vals[bound:-bound,bound:-bound]
        ssim_vals = ssim_vals[bound:-bound,bound:-bound]

        psnr_std = np.std(psnr_vals[psnr_vals > 0])
        psnr_mean = np.mean(psnr_vals[psnr_vals > 0])
        ssim_std = np.std(ssim_vals[ssim_vals > 0])
        ssim_mean = np.mean(ssim_vals[ssim_vals > 0])

        info = {'local_psnr':psnr_vals, 'local_ssim': ssim_vals, 'psnr_std': psnr_std,'psnr_mean':psnr_mean,'ssim_std':ssim_std,'ssim_mean':ssim_mean}

        return info

    #EVALUATE A PARTICULAR MODEL USING FINAL G FUNCTION
    #NOT MADE YET
    def evaluate_baseline(self,lr,hr):
        if self.model == 'RCAN':
            lr = lr * 255.0
            hr = hr * 255.0

        if self.model == 'bicubic':
            h,w,_ = hr.shape
            dim = (w,h)
            SR = cv2.resize(lr,dim, interpolation=cv2.INTER_CUBIC)
        else:
            img = torch.FloatTensor(lr).to(self.device)
            lr_img = img.permute((2,0,1)).unsqueeze(0)
            self.SRmodels[0].eval()
            SR = self.SRmodels[0](lr_img)
            SR = SR.squeeze(0).permute(1,2,0).data.cpu().numpy()

        if self.model == 'ESRGAN':
            psnr,ssim = self.getstats(SR,hr)
        elif self.model == 'RCAN':
            hr = hr / 255
            SR = SR / 255
            psnr,ssim = self.getstats(SR,hr)
        else:
            psnr,ssim = self.getstats(SR,hr)
        return psnr,ssim,SR

    #EVALUATE THE INPUT AND GET SR RESULT
    def evaluate(self,lr):
        #FORMAT THE INPUT
        h,w,d = lr.shape
        lr = torch.FloatTensor(lr).to(self.device)
        lr = lr.permute((2,0,1)).unsqueeze(0)
        SR_result = torch.zeros(1,3,h * self.upsize,w * self.upsize).to(self.device)

        #GET CHOICES AND WEIGHTED SR RESULT
        choices = self.agent.model(lr)
        for i,sisr in enumerate(self.SRmodels):
            sr = sisr(lr)
            weighted_pred = sr * choices[:,i]
            SR_result += weighted_pred
            img = sr.squeeze(0).permute(1,2,0).clamp(0,1).data.cpu().numpy()
            #plt.imshow(img)
            #plt.savefig("/home/huynshen/fig"+str(i)+".png",bbox_inches='tight')
            #plt.show()

        #FORMAT THE OUTPUT
        SR_result = SR_result.clamp(0,1)
        SR_result = SR_result.squeeze(0).permute(1,2,0).data.cpu().numpy()
        choices = choices.squeeze(0).permute(1,2,0).data.cpu().numpy()
        return SR_result,choices

    #EVALUATE THE INPUT AND GET SR RESULT
    def evaluateBounds(self,lr,hr):
        #FORMAT THE INPUT
        h,w,d = lr.shape
        lr = torch.FloatTensor(lr).to(self.device)
        hr = torch.FloatTensor(hr).to(self.device)
        lr = lr.permute((2,0,1)).unsqueeze(0)
        hr = hr.permute((2,0,1)).unsqueeze(0)
        SR_result = []

        #GET EACH SR RESULT
        bestchoice = torch.zeros(1,3,h * self.upsize,w * self.upsize).to(self.device)
        worstchoice = torch.zeros(1,3,h * self.upsize,w * self.upsize).to(self.device)
        weightedchoice = torch.zeros(1,3,h * self.upsize,w * self.upsize).to(self.device)
        choices = self.agent.model(lr)
        lowchoice, idxlow = choices.min(dim=1)
        maxchoice, idxhigh = choices.max(dim=1)
        for i,sisr in enumerate(self.SRmodels):
            sr = sisr(lr)
            SR_result.append(sr)
            masklow = idxlow == i
            maskhigh = idxhigh == i
            weightedchoice += sr * choices[:,i]
            worstchoice += sr * masklow.float()
            bestchoice += sr * maskhigh.float()

        #GET EUCLIDEAN DISTANCE FOR EACH PIXEL
        l2diff = []
        for sr in SR_result:
            pixelMSE = (sr - hr).pow(2).sqrt().mean(dim=1)
            l2diff.append(pixelMSE)
        l2diff = torch.stack(l2diff,dim=1)
        minvals,idxlow = l2diff.min(dim=1)
        maxvals,idxhigh= l2diff.max(dim=1)
        advantage = (l2diff - torch.min(l2diff)) / torch.max(l2diff)

        # create lower and upper bound mask for visual
        upperboundmask = torch.zeros(l2diff.shape)
        lowerboundmask = torch.zeros(l2diff.shape)
        for i in range(len(self.SRmodels)):
            upperboundmask[:,i] = (idxlow == i).float()
            lowerboundmask[:,i] = (idxhigh == i).float()

        #GET LOWER AND UPPER BOUND IMAGE
        lowerboundImg = torch.zeros(1,3,h * self.upsize,w * self.upsize).to(self.device)
        upperboundImg = torch.zeros(1,3,h * self.upsize,w * self.upsize).to(self.device)
        for i in range(len(SR_result)):
            masklow = idxlow == i
            maskhigh = idxhigh == i
            lowerboundImg += masklow.float().unsqueeze(1) * SR_result[i]
            upperboundImg += maskhigh.float().unsqueeze(1) * SR_result[i]

        #FORMAT THE OUTPUT
        diff = l2diff.clamp(0,1).squeeze(0).permute(1,2,0).data.cpu().numpy()
        lowerboundImg = lowerboundImg.clamp(0,1).squeeze(0).permute(1,2,0).data.cpu().numpy()
        upperboundImg = upperboundImg.clamp(0,1).squeeze(0).permute(1,2,0).data.cpu().numpy()
        bestchoice = bestchoice.clamp(0,1).squeeze(0).permute(1,2,0).data.cpu().numpy()
        worstchoice = worstchoice.clamp(0,1).squeeze(0).permute(1,2,0).data.cpu().numpy()
        weightedchoice = weightedchoice.clamp(0,255).squeeze(0).permute(1,2,0).data.cpu().numpy()
        choices = choices.clamp(0,1).squeeze(0).permute(1,2,0).data.cpu().numpy()
        idxlow = idxlow.squeeze(0).data.cpu().numpy()
        idxhigh = idxhigh.squeeze(0).data.cpu().numpy()

        info = {'HR': hr,'best': bestchoice, 'worst': worstchoice, 'weighted': weightedchoice,'lower': lowerboundImg, 'upper': upperboundImg, 'choices': choices, 'idxlow': idxlow, 'idxhigh': idxhigh, 'upperboundmask': upperboundmask, 'lowerboundmask': lowerboundmask, 'advantage': advantage, 'diff': diff}

        return info

    #GATHER STATISTICS FROM SR RESULT AND GROUND TRUTH
    def getstats(self,sr,hr):
        sr = (sr * 255).clip(0,255)
        hr = hr * 255
        psnr,ssim = util.calc_metrics(hr,sr,crop_border=self.upsize)
        return psnr,ssim

    #TEST A MODEL ON ALL DATASETS
    def validateSet5(self,save=False,quick=False):
        scores = {}
        [model.eval() for model in self.SRmodels]
        self.agent.model.eval()
        self.validationsets = ['Starfish']

        for vset in self.validationsets:
            scores[vset] = []
            HR_dir = os.path.join(self.hr_rootdir,vset,'x'+str(self.upsize))
            LR_dir = os.path.join(os.path.join(self.lr_rootdir,vset),self.resfolder)

            #SORT THE HR AND LR FILES IN THE SAME ORDER
            HR_files = [os.path.join(HR_dir, f) for f in os.listdir(HR_dir)]
            LR_files = [os.path.join(LR_dir, f) for f in os.listdir(LR_dir)]
            HR_files.sort()
            LR_files.sort()
            HR_files.reverse()
            LR_files.reverse()

            #APPLY SISR ON EACH LR IMAGE AND GATHER RESULTS
            for hr_file,lr_file in zip(HR_files,LR_files):
                hr = imageio.imread(hr_file) / 255.0
                lr = imageio.imread(lr_file) / 255.0

                #EVALUATE AND GATHER STATISTICS
                selection_details = self.evaluateBounds(lr,hr)
                psnr_low, ssim_low = self.getstats(selection_details['lower'],hr)
                psnr_high,ssim_high= self.getstats(selection_details['upper'],hr)
                psnr_best,ssim_best= self.getstats(selection_details['best'],hr)
                psnr_worst,ssim_worst = self.getstats(selection_details['worst'],hr)
                choices = selection_details['choices']
                sr = selection_details['weighted']
                psnr,ssim = self.getstats(sr,hr)

                selection_details['file'] = os.path.basename(lr_file)
                print(f"low mse: {psnr_low:.3f}/{ssim_low:.3f} | high mse: {psnr_high:.3f}/{ssim_high:.3f} | best choice: {psnr_best:.3f}/{ssim_best:.3f} | worst choice: {psnr_worst:.3f}/{ssim_worst:.3f} | psnr/ssim: {psnr:.3f}/{ssim:.3f} | {selection_details['file']}")
                scores[vset].append([psnr,ssim,psnr_low,ssim_low,psnr_high,ssim_high,psnr_best,ssim_best,psnr_worst,ssim_worst])

                #OPTIONAL THINGS TO DO
                if quick: break
                if save:
                    #save info for each file tested
                    for method in ['best','worst','weighted','lower','upper','choices']:
                        filename = os.path.join('runs',method + '_' + os.path.basename(lr_file))
                        imageio.imwrite(filename,selection_details[method].astype(np.uint8))
                        if method == 'choices':
                            plt.hist(selection_details[method].flatten() / 255.0,bins=100,range=(0,1))
                            plt.savefig(filename[:-4] + '_hist.png')
                            plt.clf()

            mu_psnr = np.mean(np.array(scores[vset])[:,0])
            mu_ssim = np.mean(np.array(scores[vset])[:,1])
            mu_psnr_low = np.mean(np.array(scores[vset])[:,2])
            mu_ssim_low = np.mean(np.array(scores[vset])[:,3])
            mu_psnr_high = np.mean(np.array(scores[vset])[:,4])
            mu_ssim_high = np.mean(np.array(scores[vset])[:,5])
            mu_psnr_best = np.mean(np.array(scores[vset])[:,6])
            mu_ssim_best = np.mean(np.array(scores[vset])[:,7])
            mu_psnr_worst = np.mean(np.array(scores[vset])[:,8])
            mu_ssim_worst = np.mean(np.array(scores[vset])[:,9])
            print( "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
                    )
            print(f"MEANS: OPTIMAL: {mu_psnr_low:.3f}/{mu_ssim_low:.3f} | WORST: {mu_psnr_high:.3f}/{mu_ssim_high:.3f} | best choice {mu_psnr_best:.3f}/{mu_ssim_worst:.3f} | worst choice {mu_psnr_worst:.3f}/{mu_ssim_worst:.3f} | psnr/ssim: {mu_psnr:.3f}/{mu_ssim:.3f}")

        return mu_psnr,mu_ssim,selection_details

    #TEST A MODEL ON ALL DATASETS
    def validate_baseline(self,save=True,quick=False):
        scores = {}
        [model.eval() for model in self.SRmodels]

        for vset in self.validationsets:
            scores[vset] = []
            HR_dir = os.path.join(self.hr_rootdir,vset,'x'+str(self.upsize))
            LR_dir = os.path.join(os.path.join(self.lr_rootdir,vset),self.resfolder)

            #SORT THE HR AND LR FILES IN THE SAME ORDER
            HR_files = [os.path.join(HR_dir, f) for f in os.listdir(HR_dir)]
            LR_files = [os.path.join(LR_dir, f) for f in os.listdir(LR_dir)]
            HR_files.sort()
            LR_files.sort()
            HR_files.reverse()
            LR_files.reverse()

            #APPLY SISR ON EACH LR IMAGE AND GATHER RESULTS
            for hr_file,lr_file in zip(HR_files,LR_files):
                hr = imageio.imread(hr_file) / 255.0
                lr = imageio.imread(lr_file) / 255.0

                #EVALUATE AND GATHER STATISTICS
                psnr,ssim,sr = self.evaluate_baseline(lr,hr)

                info = {}
                info['LRimg'] = lr
                info['HRimg'] = hr
                info['SRimg'] = sr
                info['psnr'] = psnr
                info['ssim'] = ssim

                print(lr_file,psnr,ssim)
                scores[vset].append([psnr,ssim])
                if quick: break

                #save info for each file tested
                if save:
                    filename = os.path.join('runs',vset + os.path.basename(hr_file)+'.mat')
                    info['LR_DIR'] = lr_file
                    info['HR_DIR'] = hr_file
                    sio.savemat(filename,info)
                    outpath = './'
                    for path in ['../../RCAN_TestCode/SR','BI',self.name,vset,'x' + str(self.upsize)]:
                        outpath = os.path.join(outpath,path)
                        if not os.path.exists(outpath): os.mkdir(outpath)
                    outpath = os.path.splitext(os.path.join(outpath, os.path.basename(hr_file)))[0] +'_' + self.name + '_x' + str(self.upsize) + '.png'
                    #imageio.imwrite(outpath,info['SRimg'].astype(np.uint8))

            mu_psnr = np.mean(np.array(scores[vset])[:,0])
            mu_ssim = np.mean(np.array(scores[vset])[:,1])
            print(vset + ' scores',mu_psnr,mu_ssim)

        return mu_psnr,mu_ssim,info


    #TEST A MODEL ON ALL DATASETS
    def validate(self,save=True,quick=False):
        scores = {}
        [model.eval() for model in self.SRmodels]
        self.agent.model.eval()

        for vset in self.validationsets:
            scores[vset] = []
            HR_dir = os.path.join(self.hr_rootdir,vset,'x'+str(self.upsize))
            LR_dir = os.path.join(os.path.join(self.lr_rootdir,vset),self.resfolder)

            #SORT THE HR AND LR FILES IN THE SAME ORDER
            HR_files = [os.path.join(HR_dir, f) for f in os.listdir(HR_dir)]
            LR_files = [os.path.join(LR_dir, f) for f in os.listdir(LR_dir)]
            HR_files.sort()
            LR_files.sort()
            HR_files.reverse()
            LR_files.reverse()

            #APPLY SISR ON EACH LR IMAGE AND GATHER RESULTS
            for hr_file,lr_file in zip(HR_files,LR_files):
                hr = imageio.imread(hr_file) / 255.0
                lr = imageio.imread(lr_file) / 255.0

                #EVALUATE AND GATHER STATISTICS
                testing_detail = self.evaluateBounds(lr,hr)
                psnr,ssim = self.getstats(testing_detail['weighted'],hr)
                upperpsnr,upperssim = self.getstats(testing_detail['lower'],hr)
                lowerpsnr,lowerssim = self.getstats(testing_detail['upper'],hr)

                info = {}
                info['diff'] = testing_detail['diff']
                info['LRimg'] = lr
                info['HRimg'] = hr
                info['psnr'] = psnr
                info['ssim'] = ssim
                info['upperpsnr'] = upperpsnr
                info['upperssim'] = upperssim
                info['lowerpsnr'] = lowerpsnr
                info['lowerssim'] = lowerssim
                info['SRimg'] = testing_detail['weighted']
                info['assignment'] = testing_detail['choices']
                info['lower'] = testing_detail['upper']
                info['upper'] = testing_detail['lower']
                info['uppermask'] = testing_detail['idxlow']
                info['lowermask'] = testing_detail['idxhigh']

                print(lr_file,psnr,ssim)
                scores[vset].append([psnr,ssim])
                if quick: break

                #save info for each file tested
                if save:
                    filename = os.path.join('runs',vset + os.path.basename(hr_file)+'.mat')
                    info['LR_DIR'] = lr_file
                    info['HR_DIR'] = hr_file
                    sio.savemat(filename,info)
                    outpath = './'
                    for path in ['../../RCAN_TestCode/SR','BI',self.name,vset,'x' + str(self.upsize)]:
                        outpath = os.path.join(outpath,path)
                        if not os.path.exists(outpath): os.mkdir(outpath)
                    outpath = os.path.splitext(os.path.join(outpath, os.path.basename(hr_file)))[0] +'_' + self.name + '_x' + str(self.upsize) + '.png'
                    #imageio.imwrite(outpath,info['SRimg'].astype(np.uint8))

            mu_psnr = np.mean(np.array(scores[vset])[:,0])
            mu_ssim = np.mean(np.array(scores[vset])[:,1])
            print(vset + ' scores',mu_psnr,mu_ssim)

        return mu_psnr,mu_ssim,info

    #TEST A MODEL ON ALL DATASETS
    def validate_ensemble(self,save=True,quick=False):
        scores = {}
        [model.eval() for model in self.SRmodels]
        self.agent.model.eval()

        for vset in self.validationsets:
            scores[vset] = []
            HR_dir = os.path.join(self.hr_rootdir,vset,'x'+str(self.upsize))
            LR_dir = os.path.join(os.path.join(self.lr_rootdir,vset),self.resfolder)

            #SORT THE HR AND LR FILES IN THE SAME ORDER
            HR_files = [os.path.join(HR_dir, f) for f in os.listdir(HR_dir)]
            LR_files = [os.path.join(LR_dir, f) for f in os.listdir(LR_dir)]
            HR_files.sort()
            LR_files.sort()
            HR_files.reverse()
            LR_files.reverse()

            #APPLY SISR ON EACH LR IMAGE AND GATHER RESULTS
            for hr_file,lr_file in zip(HR_files,LR_files):
                hr = imageio.imread(hr_file) / 255.0
                lr = imageio.imread(lr_file) / 255.0

                sr1,c1 = self.evaluate(lr)
                sr2,c2 = self.evaluate(np.rot90(lr,1).copy())
                sr3,c3 = self.evaluate(np.rot90(lr,2).copy())
                sr4,c4 = self.evaluate(np.rot90(lr,3).copy())
                sr5,c5 = self.evaluate(np.flip(lr,axis=1).copy())
                sr6,c6 = self.evaluate(np.rot90(np.flip(lr,axis=1),1).copy())
                sr7,c7 = self.evaluate(np.rot90(np.flip(lr,axis=1),2).copy())
                sr8,c8 = self.evaluate(np.rot90(np.flip(lr,axis=1),3).copy())

                sr1 = sr1
                sr2 = np.rot90(sr2,3)
                sr3 = np.rot90(sr3,2)
                sr4 = np.rot90(sr4,1)
                sr5 = np.flip(sr5,axis=1)
                sr6 = np.flip(np.rot90(sr6,3),axis=1)
                sr7 = np.flip(np.rot90(sr7,2),axis=1)
                sr8 = np.flip(np.rot90(sr8,1),axis=1)
                c1 = c1
                c2 = np.rot90(c2,3)
                c3 = np.rot90(c3,2)
                c4 = np.rot90(c4,1)
                c5 = np.flip(c5,axis=1)
                c6 = np.flip(np.rot90(c6,3),axis=1)
                c7 = np.flip(np.rot90(c7,2),axis=1)
                c8 = np.flip(np.rot90(c8,1),axis=1)

                #EVALUATE AND GATHER STATISTICS
                sr = (sr1 + sr2 + sr3 + sr4 + sr5 + sr6 + sr7 + sr8) / 8.0
                choices = (c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8) / 8.0
                psnr,ssim = self.getstats(sr,hr)

                info = {}
                info['LRimg'] = lr
                info['HRimg'] = hr
                info['psnr'] = psnr
                info['ssim'] = ssim
                info['SRimg'] = sr
                info['assignment'] = choices
                print(lr_file,psnr,ssim)
                scores[vset].append([psnr,ssim])
                if quick: break

                #save info for each file tested
                if save:
                    filename = os.path.join('runs',vset + os.path.basename(hr_file)+'.mat')
                    info['LRimg'] = lr
                    info['HRimg'] = hr
                    info['LR_DIR'] = lr_file
                    info['HR_DIR'] = hr_file
                    sio.savemat(filename,info)
                    outpath = './'
                    for path in ['../../RCAN_TestCode/SR','BI',self.name,vset,'x' + str(self.upsize)]:
                        outpath = os.path.join(outpath,path)
                        if not os.path.exists(outpath): os.mkdir(outpath)
                    outpath = os.path.splitext(os.path.join(outpath, os.path.basename(hr_file)))[0] +'_' + self.name + '_x' + str(self.upsize) + '.png'
                    imageio.imwrite(outpath,info['SRimg'].astype(np.uint8))

            mu_psnr = np.mean(np.array(scores[vset])[:,0])
            mu_ssim = np.mean(np.array(scores[vset])[:,1])
            print(vset + ' scores',mu_psnr,mu_ssim)

        return mu_psnr,mu_ssim,info

####################################################################################################
####################################################################################################
####################################################################################################
if __name__ == '__main__':
    ####################################################################################################
    ####################################################################################################
    with torch.no_grad():
        testing_regime = Tester()
        if args.getbounds:
            testing_regime.validateSet5()
        elif args.ensemble:
            testing_regime.validate_ensemble()
        elif args.evaluate:
            testing_regime.validate()
        elif args.baseline:
            testing_regime.validate_baseline()
        elif args.viewM:

            patchinfo = np.load(args.patchinfo)
            M = testing_regime.agent.getM()
            data = M.detach().cpu().numpy()
            data = data[:patchinfo[0],:]

            print(data[0])
            np.mean(data)
            np.std(data)
            plt.hist(data,bins='auto')
            plt.title("histogram of the weight data")
            plt.show()

        elif args.baseline:
            if args.hrimg != "":
                lrimg = imageio.imread(args.lrimg)
                hrimg = imageio.imread(args.hrimg)
                psnr,ssim,SR = testing_regime.evaluate_baseline(lrimg,hrimg)                     #evaluate the low res image and get testing metrics
                localinfo = testing_regime.getLocalScore(SR,hrimg)

                print("PSNR SCORE: {:.4f}".format(psnr) )
                print("SSIM SCORE: {:.4f}".format(ssim) )
                print("local psnr mu/std: {:.4f} / {:.4f}".format(localinfo['psnr_mean'],localinfo['psnr_std']))
                print("local ssim mu/std: {:.4f} / {:.4f}".format(localinfo['ssim_mean'],localinfo['ssim_std']))

                cv2.imshow('local psnr',((np.clip(localinfo['local_psnr'],0,50) / 50) * 255).astype(np.uint8))
                cv2.imshow('local ssim',(localinfo['local_ssim'] * 255).astype(np.uint8))
                cv2.imshow('Low Res',cv2.cvtColor(lrimg,cv2.COLOR_BGR2RGB))
                cv2.imshow('High Res',cv2.cvtColor(hrimg,cv2.COLOR_BGR2RGB))
                cv2.imshow('Super Res',cv2.cvtColor(SR,cv2.COLOR_BGR2RGB))
                cv2.waitKey(0)


        elif args.testbasic:
            if args.hrimg != "":
                lrimg = imageio.imread(args.lrimg) / 255.0
                hrimg = imageio.imread(args.hrimg)
                srimg, choices = testing_regime.evaluate(lrimg)                     #evaluate the low res image and get testing metrics
                quit()

                #choice = testing_regime.getPatchChoice(info)             #get colored mask on k model decisions
                #localinfo = testing_regime.getLocalScore(info['SRimg'],hrimg)

                print("PSNR SCORE: {:.4f}".format(psnr) )
                print("SSIM SCORE: {:.4f}".format(ssim) )
                print("local psnr mu/std: {:.4f} / {:.4f}".format(localinfo['psnr_mean'],localinfo['psnr_std']))
                print("local ssim mu/std: {:.4f} / {:.4f}".format(localinfo['ssim_mean'],localinfo['ssim_std']))

                cv2.imshow('local psnr',((np.clip(localinfo['local_psnr'],0,50) / 50) * 255).astype(np.uint8))
                cv2.imshow('local ssim',(localinfo['local_ssim'] * 255).astype(np.uint8))
                cv2.imshow('Choice Mask',cv2.cvtColor(choice,cv2.COLOR_BGR2RGB))
                cv2.imshow('Low Res',cv2.cvtColor(lrimg,cv2.COLOR_BGR2RGB))
                cv2.imshow('High Res',cv2.cvtColor(hrimg,cv2.COLOR_BGR2RGB))
                cv2.imshow('Super Res',cv2.cvtColor(info['SRimg'],cv2.COLOR_BGR2RGB))
                cv2.waitKey(0)

            elif args.lrimg != "":
                lrimg = imageio.imread(args.lrimg) / 255.0
                srimg, choices = testing_regime.evaluate(lrimg)                     #evaluate the low res image and get testing metrics
                plt.imshow(srimg)
                plt.show()
                plt.imsave(args.lrimg + '.SRx4ensemble.png',srimg)
                quit()
                print('your life is nothing')
            else:
                print('your life is nothing')

