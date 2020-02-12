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
    def __init__(self,agent=None,SRmodels=None,args=args,testset=['Set5','Set14','B100']):
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
        self.args=args
        self.hr_rootdir = os.path.join(args.dataroot,'HR')
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
                model = arch.RRDBNet(3,3,64,23,gc=32)
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
        if self.model == 'ESRGAN': lr = lr * 1.0 / 255.0
        img = torch.FloatTensor(lr).to(self.device)
        lr_img = img.permute((2,0,1)).unsqueeze(0)

        self.SRmodels[0].eval()
        SR = self.SRmodels[0](lr_img)
        SR = SR.squeeze(0).permute(1,2,0).data.cpu().numpy()
        if self.model == 'ESRGAN':
            SR = np.clip(SR * 255.0,0,255)
            psnr,ssim = util.calc_metrics(hr,SR,4)
        elif self.model == 'RCAN':
            SR = np.clip(SR,0,255)
            psnr,ssim = util.calc_metrics(hr,SR,4)

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

        #FORMAT THE OUTPUT
        SR_result = SR_result.clamp(0,255)
        SR_result = SR_result.squeeze(0).permute(1,2,0).data.cpu().numpy()
        choices = choices.squeeze(0).permute(1,2,0).data.cpu().numpy()

        return SR_result,choices

    #EVALUATE THE INPUT AND GET SR RESULT
    def evaluateBounds(self,lr,hr):
        #FORMAT THE INPUT
        h1,w1,d1 = hr.shape
        LR,HR,info = util.getTrainingPatches(lr,hr,self.args,transform=False)
        h2,w2 = info

        LR = LR.to(self.device)
        HR = HR.to(self.device)

        #GET EACH SR RESULT
        choices = self.agent.model(LR)
        maxval,maxarg = choices.max(dim=1)
        minval,minarg = choices.min(dim=1)

        sisrs = []
        l1 = []
        for i, sisr in enumerate(self.SRmodels):
            sr = sisr(LR)
            sisrs.append(sr)
            l1.append(torch.abs(sr - HR).mean(dim=1).mean(dim=1).mean(dim=1))
        l1diff = torch.stack(l1,dim=1)
        sisrs = torch.stack(sisrs,dim=1)
        mask = torch.zeros(sisrs.shape).to(self.device)
        mask[:,0,0] += 255
        mask[:,1,1] += 255
        mask[:,2,2] += 255

        _,optimal_idx = l1diff.min(dim=1)
        _,worst_idx = l1diff.max(dim=1)
        sr_opt = sisrs[torch.arange(sisrs.size(0)),optimal_idx]

        # GATHER SUPER RESOLUTION BASED ON CHOICES
        optimal = sisrs[torch.arange(sisrs.size(0)),optimal_idx]
        worst = sisrs[torch.arange(sisrs.size(0)),worst_idx]
        sr = sisrs[torch.arange(sisrs.size(0)),maxarg]
        minsisr = sisrs[torch.arange(sisrs.size(0)),minarg]
        weight = torch.zeros(sr.shape).to(self.device)
        for i,w in enumerate(choices):
            for j,c in enumerate(w):
                weight[i,j] = c

        # GET DECISION MASKS
        optimalchoice = mask[torch.arange(sisrs.size(0)),optimal_idx]
        worstchoice = mask[torch.arange(sisrs.size(0)),worst_idx]
        maxchoice = mask[torch.arange(sisrs.size(0)),maxarg]
        minchoice = mask[torch.arange(sisrs.size(0)),minarg]

        # RECOMBINE RESULTS
        optimal = util.recombine(optimal,h1,w1,h2,w2)
        worst = util.recombine(worst,h1,w1,h2,w2)
        sr = util.recombine(sr,h1,w1,h2,w2)
        minsisr = util.recombine(minsisr,h1,w1,h2,w2)
        optimalchoice = util.recombine(optimalchoice,h1,w1,h2,w2)
        worstchoice = util.recombine(worstchoice,h1,w1,h2,w2)
        maxchoice = util.recombine(maxchoice,h1,w1,h2,w2)
        minchoice = util.recombine(minchoice,h1,w1,h2,w2)
        mask = util.recombine(weight,h1,w1,h2,w2)

        #FORMAT THE OUTPUT
        optimal = optimal.clip(0,255)
        worst = worst.clip(0,255)
        sr = sr.clip(0,255)
        minsisr = minsisr.clip(0,255)
        maxchoice = maxchoice.clip(0,255)
        minchoice = minchoice.clip(0,255)
        worstchoice = worstchoice.clip(0,255)
        optimalchoice = optimalchoice.clip(0,255)
        mask = mask * 255.0

        info = {'HR': hr,'min': minsisr, 'max': sr, 'worst': worst, 'optimal': optimal, 'optimalchoice': optimalchoice, 'worstchoice': worstchoice, 'maxchoice': maxchoice, 'minchoice': minchoice, 'choices': choices, 'mask': mask}

        return info

    #GATHER STATISTICS FROM SR RESULT AND GROUND TRUTH
    def getstats(self,sr,hr):
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
            HR_dir = os.path.join(self.hr_rootdir,vset)
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
                hr = imageio.imread(hr_file)
                lr = imageio.imread(lr_file)

                #EVALUATE AND GATHER STATISTICS
                selection_details = self.evaluateBounds(lr,hr)
                psnr_best, ssim_best = self.getstats(selection_details['max'],hr)
                psnr_low, ssim_low = self.getstats(selection_details['min'],hr)
                psnr_worst, ssim_worst = self.getstats(selection_details['worst'],hr)
                psnr_optimal, ssim_optimal = self.getstats(selection_details['optimal'],hr)
                choices = selection_details['choices']

                selection_details['file'] = os.path.basename(lr_file)
                print(f"worst mse: {psnr_worst:.3f}/{ssim_worst:.3f} | optimal mse: {psnr_optimal:.3f}/{ssim_optimal:.3f} | best choice: {psnr_best:.3f}/{ssim_best:.3f} | bad choice: {psnr_low:.3f}/{ssim_low:.3f} | {selection_details['file']}")
                scores[vset].append([psnr_worst,ssim_worst,psnr_optimal,ssim_optimal,psnr_best,ssim_best,psnr_low,ssim_low])

                #OPTIONAL THINGS TO DO
                if quick: break
                if save:
                    #save info for each file tested
                    for method in ['min','max','worst','optimal','optimalchoice','worstchoice','maxchoice','minchoice','choices', 'mask']:
                        filename = os.path.join('runs',method + '_' + os.path.basename(lr_file))
                        imageio.imwrite(filename,selection_details[method].astype(np.uint8))
                        if method == 'choices':
                            plt.hist(selection_details[method].flatten() / 255.0,bins=100,range=(0,1))
                            plt.savefig(filename[:-4] + '_hist.png')
                            plt.clf()

            mu_psnr_worst = np.mean(np.array(scores[vset])[:,0])
            mu_ssim_worst = np.mean(np.array(scores[vset])[:,1])
            mu_psnr_optimal = np.mean(np.array(scores[vset])[:,2])
            mu_ssim_optimal = np.mean(np.array(scores[vset])[:,3])
            mu_psnr_best = np.mean(np.array(scores[vset])[:,4])
            mu_ssim_best = np.mean(np.array(scores[vset])[:,5])
            mu_psnr_low = np.mean(np.array(scores[vset])[:,6])
            mu_ssim_low = np.mean(np.array(scores[vset])[:,7])
            print( "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
                    )
            print(f"worst mse: {mu_psnr_worst:.3f}/{mu_ssim_worst:.3f} | optimal mse: {mu_psnr_optimal:.3f}/{mu_ssim_optimal:.3f} | best choice: {mu_psnr_best:.3f}/{mu_ssim_best:.3f} | bad choice: {mu_psnr_low:.3f}/{mu_ssim_low:.3f} | {selection_details['file']}")
        return mu_psnr_best,mu_ssim_best,selection_details

    #TEST A MODEL ON ALL DATASETS
    def validate(self,save=True,quick=False):
        scores = {}
        [model.eval() for model in self.SRmodels]
        self.agent.model.eval()

        for vset in self.validationsets:
            scores[vset] = []
            HR_dir = os.path.join(self.hr_rootdir,vset)
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
                hr = imageio.imread(hr_file)
                lr = imageio.imread(lr_file)

                #EVALUATE AND GATHER STATISTICS
                sr,choices = self.evaluate(lr)
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

    #TEST A MODEL ON ALL DATASETS
    def validate_ensemble(self,save=True,quick=False):
        scores = {}
        [model.eval() for model in self.SRmodels]
        self.agent.model.eval()

        for vset in self.validationsets:
            scores[vset] = []
            HR_dir = os.path.join(self.hr_rootdir,vset)
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
                hr = imageio.imread(hr_file)
                lr = imageio.imread(lr_file)

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
        elif args.viewM:

            patchinfo = np.load(args.patchinfo)
            M = testing_regime.agent.getM()
            data = M.detach().cpu().numpy()
            data = data[:patchinfo[0],:]

            print(data[0])
            np.mean(data)
            np.std(data)
            #matplotlib.use('tkagg')
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
                lrimg = imageio.imread(args.lrimg)
                hrimg = imageio.imread(args.hrimg)
                psnr,ssim,info = testing_regime.evaluate_ideal(lrimg,hrimg)                     #evaluate the low res image and get testing metrics
                choice = testing_regime.getPatchChoice(info)             #get colored mask on k model decisions
                localinfo = testing_regime.getLocalScore(info['SRimg'],hrimg)

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
                lrimg = imageio.imread(args.lrimg)
                print('your life is nothing')
            else:
                print('your life is nothing')

