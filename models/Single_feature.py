from argparse import Namespace
from collections import Counter
import csv
import gc
from itertools import product
from re import S
from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
from numpy import array, pad
import os
import pandas as pd
from tqdm import tqdm_notebook as tqdm

# FT block
class FFT1D_block(nn.Module):
    def __init__(self, modes1,modes2, num=None):
        super(FFT1D_block, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.out_channels = modes2
        self.num = num
        self.scale = (1 / (modes1*modes2))
        self.weights_FFT = nn.Parameter(self.scale * torch.rand(modes1,modes2, dtype=torch.cfloat))
        
    def fft_1D(self, x):
        if self.num:
            
            x_ft = torch.fft.fft(x,dim=1)
            x_ft = torch.einsum("bix,io->box",x_ft , self.weights_FFT)
            x = torch.fft.ifft(x_ft,dim=1, n=self.num)
            x = x.real 
        else:
            
            x_ft = torch.fft.fft(x,dim=1)
            x_ft = torch.einsum("bix,io->box",x_ft , self.weights_FFT)
            x = torch.fft.ifft(x_ft,dim=1, n=self.out_channels)
            x = x.real 
        return x
    
    def forward(self, x):
        x = self.fft_1D(x)
        return x

# SVD block
class SVD_Block_3(nn.Module):
    def __init__(self, modes1,modes2, F_channels):
        super(SVD_Block_3, self).__init__()
       
        self.W_FFT1 = FFT1D_block(modes1,modes2)
        self.weight_SVD = nn.Parameter(torch.rand(modes1, F_channels))
       
    def forward(self, x):
        u,s,v = torch.linalg.svd(x,full_matrices=False)
        wu,ws,wv = torch.linalg.svd(self.weight_SVD,full_matrices=False)
        s = ws*s
        s = torch.diag_embed(s)
      
        u = u*wu
        v = v*wv
        
        out_us = torch.matmul(u,s)
        out_svd = torch.matmul(out_us,v)
        
        out_svd = torch.sin(out_svd)

        out_1 = self.W_FFT1(x)
        out_1 = out_1 + out_svd
        return out_1



#  F-SVD   single feture pred
class F_SVD_Net(nn.Module):
    def __init__(self, channels, pred_num, F_channels):
        super(F_SVD_Net, self).__init__()

        
        self.in_expend = nn.Conv1d(1,4,1)
        self.out_expend = nn.Conv1d(4,1,1)
        self.svd_block_1 = SVD_Block_3(channels, channels, F_channels)
        self.W_FFT_out = FFT1D_block(channels,channels, pred_num)
    def forward(self, x):
        x_ = x[:, -1:, :].detach()
        x = x - x_
        x = self.in_expend(x.permute(0,2,1)).permute(0,2,1)
        out = self.svd_block_1(x)
        out = self.W_FFT_out(out)
        out = self.out_expend(out.permute(0,2,1)).permute(0,2,1)
        out = out + x_
        return out


#    FT conv block
class FFT_Conv(nn.Module):
    def __init__(self, in_channel,out_channel ):
        super(FFT_Conv, self).__init__()


        
        self.W_FFT1 = FFT1D_block(in_channel,out_channel)
        self.conv = nn.Conv1d(in_channel,out_channel,3,1,1)
        self.relu = nn.ReLU()
    def forward(self, x):
        conv_x = self.conv(x)
        out_1 = self.W_FFT1(x)
        out_1 = out_1 + self.relu(conv_x)
        return out_1



class FFT_Conv_Net(nn.Module):
    def __init__(self, channels, pred_num, F_channels):
        super(FFT_Conv_Net, self).__init__()

        
        self.channels = channels
        self.svd_block_1 = FFT_Conv(channels, pred_num)
       
        self.W_FFT_out = FFT1D_block(pred_num, pred_num, pred_num)
    def forward(self, x):
        x_ = x[:, -1:, :].detach()
        x = x - x_
        out = self.svd_block_1(x)
        out = self.W_FFT_out(out)
        out = out + x_
        return out



# SVD Conv block
class Conv_SVD(nn.Module):
    def __init__(self, channels, pred_num, F_channels):
        super(Conv_SVD, self).__init__()

        self.W_SVD = SVD_Block_3(channels, channels, F_channels)
        self.conv = nn.Conv1d(channels,channels,3,1,1)
        self.relu = nn.ReLU()
    def forward(self, x):
        conv_x = self.conv(x)
        out_1 = self.W_SVD(x)
        out_1 = self.relu(out_1) + conv_x
        return out_1


# Conv_SVD_Net
class Conv_SVD_Net(nn.Module):
    def __init__(self, channels, pred_num, F_channels):
        super(Conv_SVD_Net, self).__init__()

        self.in_expend = nn.Conv1d(1,4,1)
        self.out_expend = nn.Conv1d(4,1,1)
        self.svd_block_1 = Conv_SVD(channels, pred_num, F_channels)
        self.W_FFT_out = nn.Conv1d(channels,pred_num,3,1,1)
    def forward(self, x):
        x_ = x[:, -1:, :].detach()
        x = x - x_
        x = self.in_expend(x.permute(0,2,1)).permute(0,2,1)
        out = self.svd_block_1(x)
        out = self.W_FFT_out(out)
        out = self.out_expend(out.permute(0,2,1)).permute(0,2,1)
        out = out + x_
        return out



#   FT_Conv_matrix_Net
def rotate_matrix(h, w, k=3):
    vec = np.ones([w*k],dtype=np.float32)
   
    matrix_out = np.zeros((h*w,h*w))
    i = 0
    while (i+k+1)*w<=h*w:
        matrix_out[i,i*w:(i+k)*w] = vec
        i+=1
    for i in range(1,k-1):
        for j in range(h*w):
            if j//(i+2) == 0:
                matrix_out[-i,j:j+2]  = 1 
    
    return matrix_out


class FT_Conv_matrix(nn.Module):
    def __init__(self, channels, pred_num, F_channels):
        super(FT_Conv_matrix, self).__init__()

       
        self.W_FFT1 = FFT1D_block(channels,channels)
        self.weights_1 = nn.Parameter(torch.rand((channels*F_channels, channels*F_channels),dtype=torch.float), requires_grad= True)
        self.matrix = torch.from_numpy(rotate_matrix(channels, F_channels, k=2))
        self.matrix = torch.tensor(self.matrix,dtype=torch.float)
       
        self.relu = nn.ReLU()
    def forward(self, x):
        B,T,C = x.shape
        x_ = x.reshape(B,-1)
        #self.weights_1 = self.matrix.to(x.device)*self.weights_1
        midfeature = torch.einsum('bc, co->bo',x_, self.matrix.to(x.device)*self.weights_1)
        midfeature = midfeature.reshape(B,T,C)
        out_1 = self.W_FFT1(x)
        out_1 = out_1 + self.relu(midfeature)
        return out_1

class FT_Conv_matrix_Net(nn.Module):
    def __init__(self, channels, pred_num, F_channels):
        super(FT_Conv_matrix_Net, self).__init__()

        
        self.svd_block_1 = FT_Conv_matrix(channels, pred_num, F_channels)
        self.W_FFT_out = FFT1D_block(channels, channels, pred_num)
    def forward(self, x):
        x_ = x[:, -1:, :].detach()
        x = x - x_
        out = self.svd_block_1(x)
        out = self.W_FFT_out(out)
        out = out + x_
        return out

