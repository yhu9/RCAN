import os

import torch
import numpy as np

import utility
import model
from option import args
from utils import util
from importlib import import_module
import imageio

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

#######################################################################################################
#######################################################################################################

if checkpoint.ok:
    device = 'cuda:0'
    module = import_module('model.'+args.model.lower())
    model = module.make_model(args).to(device)
    kwargs = {}
    model.load_state_dict(torch.load(args.pre_train, **kwargs),strict=False)


    LRDIR = '../../../data/testing/LRBI/Set5/x4'
    HRDIR = '../../../data/testing/HR/Set5'
    lrdata = os.listdir(LRDIR)
    hrdata = os.listdir(HRDIR)
    lrdata.sort()
    hrdata.sort()
    psnr_scores = []
    ssim_scores = []

    #RUN ON SET 5 DATASET
    for lrname,hrname in zip(lrdata,hrdata):

        img = imageio.imread(os.path.join(LRDIR,lrname))
        hr = imageio.imread(os.path.join(HRDIR,hrname))

        img = torch.FloatTensor(img).to(device)
        img = img.permute((2,0,1)).unsqueeze(0)

        #MAKE THE INFERENCE
        model.eval()
        with torch.no_grad():
            sr = model(img)
            sr = sr.squeeze(0).permute(1,2,0).data.cpu().numpy()

        # hr = hr
        psnr,ssim = util.calc_metrics(hr,sr,4)
        psnr_scores.append(psnr)
        ssim_scores.append(ssim)
        print('psnr score: {:.4f}   | {}'.format(psnr,lrdata))

    print('mean and ssim: ',np.mean(psnr_scores),np.mean(ssim_scores))
    checkpoint.done()


