import argparse
import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from experiments.exp_ETTh import Exp_ETTh as Exp_ETTh_FFT


parser = argparse.ArgumentParser(description='FFT on ETT dataset')

# FFT_Conv_Net, F_SVD_Net, Conv_SVD_Net, FT_Conv_matrix_Net
parser.add_argument('--model_name', type=str, required=False, default='F_SVD_Net', help='model of the experiment')
### -------  dataset settings --------------
parser.add_argument('--data', type=str, required=False, default='ETTh1', choices=['ETTh1', 'ETTh2', 'ETTm1','ETTm2','weather','ECL','traffic','exchange','ILI'], help='name of dataset')
parser.add_argument('--root_path', type=str, default='datasets/ETT-data', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='location of the data file')
parser.add_argument('--features', type=str, default='M', choices=['S', 'M'], help='features S is univariate, M is multivariate')
parser.add_argument('--target', type=str, default='OT', help='target feature')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='exp/ETT_checkpoints/', help='location of model checkpoints')
parser.add_argument('--inverse', type=bool, default =False, help='denorm the output data')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')

### -------  device settings --------------
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='cuda:2',help='device ids of multile gpus')
                                                                                  
### -------  input/output length settings --------------   

parser.add_argument('--seq_len', type=int, default= 1440, help='input sequence length of SCINet encoder, look back window')
parser.add_argument('--label_len', type=int, default= 0 , help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default= 720, help='pr   ediction sequence length, horizon')
#parser.add_argument('--modes', default=336 , type=int, help='num of FFT')
parser.add_argument('--F_channels', default=7, type=int, help='num of FFT7')

                                                              
### -------  training settings --------------  
parser.add_argument('--cols', type=str, nargs='+', help='file list')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')

parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--lr', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='mae',help='loss function')
parser.add_argument('--lradj', type=int, default=1,help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--save', type=bool, default =False, help='save the output results')
#parser.add_argument('--model_name', type=str, default='FFT')
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--evaluate', type=bool, default=False)


args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[1]

#biuld a dict of all data,info about file path,output, and feature, so that we can get the info according to the same method 
data_parser = {
    'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ECL': {'data': 'ECL.csv', 'T': 'MT_320', 'M': [321, 321, 321], 'S': [1, 1, 1], 'MS': [321, 321, 1]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

torch.manual_seed(4321)  # reproducible
torch.cuda.manual_seed_all(4321)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False  # Can change it to False --> default: False
torch.backends.cudnn.enabled = True

Exp = Exp_ETTh_FFT
#Exp = Exp_ETTh

mae_ = []
maes_ = []
mse_ = []
mses_ = []

if args.evaluate:
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_bs{}'.format(args.model_name,args.data, args.features, args.seq_len, args.label_len, args.pred_len,args.lr,args.batch_size, )
    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    mae, maes, mse, mses = exp.test(setting, evaluate=True)
    print('Final mean normed mse:{:.4f},mae:{:.4f},denormed mse:{:.4f},mae:{:.4f}'.format(mse, mae, mses, maes))
else:
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_bs{}'.format(args.model_name,args.data, args.features, args.seq_len, args.label_len, args.pred_len,args.lr,args.batch_size)
    args.folder_path = 'exp/'+ args.model_name+'/' +args.model_name+ setting + '/'  #save results
    exp = Exp(args)  # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    mae, maes, mse, mses = exp.test(setting)
    print('Final mean normed mse:{:.4f},mae:{:.4f},denormed mse:{:.4f},mae:{:.4f}'.format(mse, mae, mses, maes))
    with open(args.folder_path+'setting.txt','w') as f:
        f.write('model_name:{}\n'.format(args.model_name))
        f.write('data set:{}\n'.format(args.data))
        f.write('features:{}\n'.format(args.features))
        f.write('sequence length:{}\n'.format(args.seq_len))
        f.write('label length:{}\n'.format(args.label_len))
        f.write('pred length:{}\n'.format(args.pred_len))
        #f.write('modes:{}\n'.format(args.modes))
        f.write('train_epochs:{}\n'.format(args.train_epochs))
        f.write('batch_size:{}\n'.format(args.batch_size))
        f.write('patience:{}\n'.format(args.patience))
        f.write('lr:{}\n'.format(args.lr))
        f.write('loss:{}\n'.format(args.loss))
        f.write('folder_path:{}\n'.format(args.folder_path))



