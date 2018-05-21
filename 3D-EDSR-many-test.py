import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim.lr_scheduler as lrs

import numpy as np
import cv2
import nrrd

import os
import glob
import sys

root = os.getcwd()

def PaddingData(datas,scale=4):
    row,col,cha = datas.shape
    if row%scale==0:
        pass
    else:
        padding = np.zeros((scale-(row-scale*(row//scale)),col,cha),dtype=datas.dtype)
        datas = np.concatenate((datas,padding),axis=0)
    row,col,cha = datas.shape
    if col%scale==0:
        pass
    else:
        padding = np.zeros((row,scale-(col-scale*(col//scale)),cha),dtype=datas.dtype)
        datas = np.concatenate((datas,padding),axis=1)
    row,col,cha = datas.shape
    if cha%scale==0:
        pass
    else:
        padding = np.zeros((row,col,scale-(cha-scale*(cha//scale))),dtype=datas.dtype)
        datas = np.concatenate((datas,padding),axis=2)
    return datas
##########################################################################################################################################
def default_conv(in_channelss, out_channels, kernel_size, bias=True):
    return nn.Conv3d(
        in_channelss, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class EDSR(nn.Module):
    def __init__(self,conv=default_conv):
        super(EDSR, self).__init__()
        n_feats = 64#64
        kernel_size = 3
        n_resblock = 16#16
        act = nn.ReLU(True)
        res_scale = 1
        scale = 4
        
        self.head = nn.Sequential(conv(1,n_feats,kernel_size))
        
        modules_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale) for _ in range(n_resblock)]
        self.body = nn.Sequential(*modules_body)
        
        modules_tail = [
            nn.Upsample(scale_factor=scale,mode='trilinear'),
            conv(n_feats, 1, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)
        
    def forward(self, x):
        x = x.contiguous()
        x = self.head(x) 
        
        res = self.body(x)
        res += x         
        
        x = self.tail(res)

        x = torch.squeeze(x,dim=1)
        return x 
##########################################################################################################################################

if __name__=='__main__':
    edsr = EDSR().cuda()
    edsr.load_state_dict(torch.load(os.path.join(root,'model','hcp_edsr_2_params.pkl')))
    
    for datas_path in glob.glob(os.path.join(root,'test_dataset','*.nrrd')):
        print('processing nrrd {}'.format(datas_path))
        test_data_name = datas_path.split('/')[-1][:-5]
        original_datas,original_options = nrrd.read(datas_path)
        original_shape = original_datas.shape
        print(original_datas.shape)

        cha_step = 20
        row_step = 20
        col_step = 20
        offect = 2

        original_datas = PaddingData(original_datas,scale=20).astype(np.float32)
        padding_shape = list(map(lambda x:x[0]-x[1],zip(original_datas.shape,original_shape)))
        print(original_datas.shape)

        result = np.zeros((4*original_datas.shape[0],4*original_datas.shape[1],4*original_datas.shape[2]),dtype=original_datas.dtype)

        one = np.zeros((original_datas.shape[0],original_datas.shape[1],offect))
        _padding_datas = np.concatenate((one,original_datas,one),axis=2)
        two = np.zeros((offect,_padding_datas.shape[1],_padding_datas.shape[2]))
        _padding_datas = np.concatenate((two,_padding_datas,two),axis=0)
        three = np.zeros((_padding_datas.shape[0],offect,_padding_datas.shape[2]))
        _padding_datas = np.concatenate((three,_padding_datas,three),axis=1)
        print(_padding_datas.shape)

        for cha in range(offect,original_datas.shape[-1]+offect,cha_step):
            for row in range(offect,original_datas.shape[0]+offect,row_step):
                for col in range(offect,original_datas.shape[1]+offect,col_step):
                    _original_datas = _padding_datas[row-offect:row+row_step+offect,col-offect:col+col_step+offect,cha-offect:cha+cha_step+offect]
                    print(_original_datas.shape)
                    
                    x = _original_datas.transpose((2,0,1))[np.newaxis,np.newaxis,:,:,:]
                    x =  Variable(torch.from_numpy(x)).float().cuda()
                    out = edsr(x)
                    
                    print('cha:{},row:{},col:{}'.format(cha,row,col))
                    
                    cache = out.cpu().data.numpy()[0,:,:,:].transpose((1,2,0))
                    result[(row-offect)*4:(row+row_step-offect)*4,(col-offect)*4:(col+col_step-offect)*4,(cha-offect)*4:(cha+cha_step-offect)*4] = cache[4*offect:-4*offect,4*offect:-4*offect,4*offect:-4*offect]
                    
        result = result[:-4*padding_shape[0] if padding_shape[0]>0 else -1,:-4*padding_shape[1] if padding_shape[1]>0 else -1,:-4*padding_shape[2] if padding_shape[2]>0 else -1]

        result = (65535*((result-result.min())/(result.max()-result.min()))).astype(np.uint16)
        original_options['sizes'] = [result.shape[1],result.shape[0],result.shape[2]]
        original_options['space directions'] = [['1','0','0'],['0','1','0'],['0','0','1']]
        original_options['space origin'] = ['0','0','0']
        original_options['type'] = 'unsigned_short'
        nrrd.write('result/'+test_data_name+'_result.nrrd',result,options=original_options)
