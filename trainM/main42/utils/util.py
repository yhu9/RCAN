import os
import math
from datetime import datetime
import numpy as np
from PIL import Image
import cv2

import torch

####################
# miscellaneous
####################

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('[Warning] Path [%s] already exists. Rename it to [%s]' % (path, new_name))
        os.rename(path, new_name)
    os.makedirs(path)


####################
# image convert
####################
def Tensor2np(tensor_list, rgb_range):

    def _Tensor2numpy(tensor, rgb_range):
        array = np.transpose(quantize(tensor, rgb_range).numpy(), (1, 2, 0)).astype(np.uint8)
        return array

    return [_Tensor2numpy(tensor, rgb_range) for tensor in tensor_list]


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                                [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                           [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def save_img_np(img_np, img_path, mode='RGB'):
    if img_np.ndim == 2:
        mode = 'L'
    img_pil = Image.fromarray(img_np, mode=mode)
    img_pil.save(img_path)


def quantize(img, rgb_range):
    pixel_range = 255. / rgb_range
    # return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)
    return img.mul(pixel_range).clamp(0, 255).round()


####################
# metric
####################
def calc_metrics(img1, img2, crop_border, test_Y=True,out=''):
    #GOT RID OF THESE TWO LINES CUZ I DID IT BEFOREHAND
    img1 = img1 / 255.
    img2 = img2 / 255.

    if test_Y and img1.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
        im1_in = rgb2ycbcr(img1)
        im2_in = rgb2ycbcr(img2)
    else:
        im1_in = img1
        im2_in = img2
    height, width = img1.shape[:2]
    if im1_in.ndim == 3:
        cropped_im1 = im1_in[crop_border:height-crop_border, crop_border:width-crop_border, :]
        cropped_im2 = im2_in[crop_border:height-crop_border, crop_border:width-crop_border, :]
    elif im1_in.ndim == 2:
        cropped_im1 = im1_in[crop_border:height-crop_border, crop_border:width-crop_border]
        cropped_im2 = im2_in[crop_border:height-crop_border, crop_border:width-crop_border]
    else:
        raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im1_in.ndim))

    psnr = calc_psnr(cropped_im1 * 255, cropped_im2 * 255)
    try:
        ssim = calc_ssim(cropped_im1 * 255, cropped_im2 * 255)
    except:
        ssim = 0.0
    return psnr,ssim

    #Sliding window approach to find local psnr and ssim values
    #psnr_vals = np.zeros(cropped_im1.shape)
    #ssim_vals = np.zeros(cropped_im1.shape)
    #h,w = cropped_im1.shape[:2]
    #for i in range(16,h-16,1):
    #    for j in range(16,w-16,1):
    #        img1 = cropped_im1[i-16:i+16,j-16:j+16]
    #        img2 = cropped_im2[i-16:i+16,j-16:j+16]
    #        psnr_vals[i,j] = calc_psnr(img1 * 255,img2 * 255)
    #        ssim_vals[i,j] = calc_ssim(img1 * 255,img2 * 255)

    #psnr_std = np.std(psnr_vals[psnr_vals > 0])
    #psnr_mean = np.mean(psnr_vals[psnr_vals > 0])
    #ssim_std = np.std(ssim_vals[ssim_vals > 0])
    #ssim_mean = np.mean(ssim_vals[ssim_vals > 0])

    #psnr_vals[psnr_vals > 0] -= np.min(psnr_vals[psnr_vals > 0])
    #psnr_vals /= np.max(psnr_vals)
    #psnr_vals *= 255.0
    #ssim_vals[ssim_vals > 0] -= np.min(ssim_vals[ssim_vals > 0])
    #ssim_vals /= np.max(ssim_vals)
    #ssim_vals *= 255.0
    #print(psnr_std, psnr_mean)
    #print(ssim_std, ssim_mean)

    #np.save(os.path.join('out','psnr_' + out),psnr_vals)
    #np.save(os.path.join('out','ssim_' + out),ssim_vals)

    #cv2.imshow('psnr',psnr_vals.astype(np.uint8))
    #cv2.imshow('ssim',(ssim_vals * 255).astype(np.uint8))
    #cv2.waitKey(1)

    return psnr, ssim

def calc_psnr(img1, img2):
    # img1 and img2 have range [0, 255]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    if np.any((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)) <= 0.0001: ssim_map = 0.0
    else:
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def calc_ssim(img1, img2):

    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

# translate numpy image
def translateXPos(img, d=4):
    h,w,d = img.shape

def translateYPos(img, d=4):
    h,w,d = img.shape

# rotate 4d image tensor of BxCxHxW by desired interval of 90 degrees
def rotateTensor(img, angle=90):
    if angle == 90:
        return img.transpose(2,3)
    elif angle == 180:
        return img.flip(3).flip(2)
    elif angle == 270:
        return img.transpose(2,3).flip(3).flip(2)

# mirror 4d image tensor of BxCxHxW
def mirrorTensor(img):
    return img.flip(3)

