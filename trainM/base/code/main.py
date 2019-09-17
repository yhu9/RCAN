import torch

import utility
import model
from option import args
from importlib import import_module
import imageio

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)




module = import_module('model.rcan')
model = module.make_model(args).to('cuda:1')

img = imageio.imread('../LR/LRBI/Set5/x4/baby_LRBI_x4.png')
img = torch.FloatTensor(img).to('cuda:1')
img = img.permute((2,0,1)).unsqueeze(0)
print(img.shape)
quit()
#img = torch.zeros((1,3,60,60)).float().to('cuda:1')
model(img)

quit()



if checkpoint.ok:
    model = model.Model(args, checkpoint)



    #checkpoint.done()




