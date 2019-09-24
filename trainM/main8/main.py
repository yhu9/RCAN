import torch

import utility
import model
from option import args
from importlib import import_module
import imageio

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    module = import_module('model.'+args.model.lower())
    model = module.make_model(args).to('cuda:0')
    kwargs = {}
    model.load_state_dict(torch.load(args.pre_train, **kwargs),strict=False)

    print('hello')
    img = imageio.imread('../LR/LRBI/Set5/x4/bird_LRBI_x4.png')
    img = torch.FloatTensor(img).to('cuda:1')
    img = img.permute((2,0,1)).unsqueeze(0)
    img = torch.zeros((1,3,60,60)).float().to('cuda:1')
    print('hello')

    model.eval()
    sr = model(img)
    print('hello')

    checkpoint.done()




