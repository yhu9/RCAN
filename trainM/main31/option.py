import argparse
import template

parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

#MASA'S TRAINING SPECIFICATIONS
parser.add_argument("--srmodel_path",default="../model/RCAN_BIX4.pt", help='Path to the SR model')
parser.add_argument("--batch_size",default=32, type=int,help='Batch Size')
parser.add_argument("--gamma",default=.9, help='Gamma Value for RL algorithm')
parser.add_argument("--eps_start",default=.90, help='Epsilon decay start value')
parser.add_argument("--eps_end",default=0.10, help='Epsilon decay end value')
parser.add_argument("--eps_decay",default=10000, help='Epsilon decay fractional step size')
parser.add_argument("--target_update",default=20, help='Target network update time')
parser.add_argument("--action_space",default=3,type=int, help='Action Space size')
parser.add_argument("--memory_size",default=100000, help='Memory Size')
parser.add_argument("--model_dir",default="",help='specify if restarting training, or doing testing',required=False)
parser.add_argument("--training_lrpath",default="../../../data/DIV2K_train_LR_bicubic/X4")
#parser.add_argument("--training_lrpath",default="LR")
parser.add_argument("--training_hrpath",default="../../../data/DIV2K_train_HR")
parser.add_argument("--testing_path",default="../../../data/DIV2K_train_LR_bicubic/X4")
parser.add_argument("--patchsize",default=16,type=int,help="patch size to super resolve")
parser.add_argument("--loadagent",default=False, action='store_const',const=True)
parser.add_argument("--learning_rate",default=0.01,help="Learning rate of Super Resolution Models")
parser.add_argument("--upsize", default=4,help="Upsampling size of the network")
parser.add_argument("--random",default=False,action='store_const',const=True,help='Set Super resolution models to random state')
parser.add_argument("--gen_patchinfo",default=False,action='store_const',const=True)
parser.add_argument("--device",default='cuda:0',help='set device to train on')
parser.add_argument("--finetune",default=True,action='store_const',const=False)
parser.add_argument("--name", required=True, help='Name to give this training session')
parser.add_argument("--ESRGAN_PATH",default="../model/RRDB_PSNR_ESRGAN_x4.pth",help='path to ESRGAN')
parser.add_argument("--step",default=0,type=int,help='determine where to start training at')

#MASA'S TESTING SPECIFICATIONS
parser.add_argument("--dataroot",default="../../../data/testing")
parser.add_argument("--down_method",default="BI",help='method of downsampling. [BI|BD]')
parser.add_argument("--evaluate",default=False,action='store_const',const=True,help='Evaluate a model with validation sets')
parser.add_argument("--view",default=False,action='store_const',const=True,help='View the agent decisions')
parser.add_argument("--testbasic",default=False,action='store_const',const=True,help='Basic test on a single lr image without corresponding hr image, and metrics')
parser.add_argument("--baseline",default=False,action='store_const',const=True,help='Basic test on a single lr image with corresponding hr image using baseline model')
parser.add_argument("--viewM",default=False,action='store_const',const=True,help='view the weight matrix M and its distributions')
parser.add_argument("--lrimg",default="",help="define low resolution image to test on")
parser.add_argument("--hrimg",default="",help="define hr resolution image to compare with")
parser.add_argument("--ensemble",default=False,action='store_const',const=True,help='apply ensemble testing method')
parser.add_argument("--getbounds",default=False,action='store_const',const=True,help='get lower and upper bound of selection network')
parser.add_argument("--viewassignment",default=False,action='store_const',const=True,help='view assignment module as well as variance of k as image map')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=3,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='/home/yulun/data/PyTorch/npy',
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='DIV2K',
                    help='test dataset name')
parser.add_argument('--benchmark_noise', action='store_true',
                    help='use noisy benchmark sets')
parser.add_argument('--n_train', type=int, default=800,
                    help='number of training set')
parser.add_argument('--n_val', type=int, default=10,
                    help='number of validation set')
parser.add_argument('--offset_val', type=int, default=800,
                    help='validation index offest')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--scale', default='4',
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=192,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=1,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--noise', type=str, default='.',
                    help='Gaussian noise std.')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')

# Model specifications
parser.add_argument('--model', default='RCAN',
                    help='model name')

parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='../model/RCAN_BIX4.pt',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=20,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=3000,
                    help='number of epochs to train')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=200,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e6',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save', type=str, default='RCAN',
                    help='file name to save')
parser.add_argument('--load', type=str, default='.',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--print_model', action='store_true',
                    help='print model')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')

# New options
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')
parser.add_argument('--testpath', type=str, default='../test/DIV2K_val_LR_our',
                    help='dataset directory for testing')
parser.add_argument('--testset', type=str, default='Set5',
                    help='dataset name for testing')
parser.add_argument('--degradation', type=str, default='BI',
                    help='degradation model: BI, BD')


args = parser.parse_args()
template.set_template(args)

args.scale = list(map(lambda x: int(x), args.scale.split('+')))

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
