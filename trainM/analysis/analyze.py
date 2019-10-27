# native imports
import glob
import os

# opensource imports
import torch
import torchvision.models as models
import imageio

# local imports
from utils import util
from option import args     #COMMAND LINE ARGUMENTS VIEW option.py file

########################################################################

HRPATH = "../../../data/DIV2K_train_HR"
LRPATH = "../../../data/DIV2K_train_LR_bicubic/X4"
LR_PATHS = glob.glob(os.path.join(LRPATH,"*"))
HR_PATHS = glob.glob(os.path.join(HRPATH,"*"))
LR_PATHS.sort()
HR_PATHS.sort()
VGG = models.vgg19(pretrained=True)
VGG.to(args.device)

# forward function using vgg19 network just to get latent vectors
def forward(x,VGG=VGG):
    VGG.eval()
    with torch.no_grad():
        x = VGG.features(x)
        x = VGG.avgpool(x)
        x = torch.flatten(x,1)
        return x

# run feature extraction on all low resolution image patches
data = []
label = []
indices = list(range(len(HR_PATHS)))
for n, idx in enumerate(indices):
    hr_path = HR_PATHS[idx]
    lr_path = LR_PATHS[idx]

    LR = imageio.imread(lr_path)
    HR = imageio.imread(hr_path)

    LR,HR,_ = util.getTrainingPatches(LR,HR,args)
    patch_ids = list(range(len(LR)))
    for i in range(0,len(LR),32):
        labels = torch.Tensor(patch_ids[i:i+32]).long()

        batch = LR[labels,:,:,:]
        batch = batch.to(args.device)
        features = forward(batch)

