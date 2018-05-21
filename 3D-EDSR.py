import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim.lr_scheduler as lrs

import numpy as np
import cv2
import nrrd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import glob
import random

root = os.getcwd()
if_restore = True #是否重载入模型

#对数据进行padding，是否padding的判断原则是数据的三个维度能否整除scale这个数，不能整除的话会padding至可以整除，否则的话不做处理
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

#三维数据四倍下采样
def ThreeDDown(datas,scale=4):
    print(datas.shape)
    
    row,col,cha = datas.shape
    datas = cv2.resize(datas,(col//2,row//2),interpolation=cv2.INTER_CUBIC)
    print(datas.shape)
    
    datas = datas.transpose((1,2,0))
    row,col,cha = datas.shape
    datas = cv2.resize(datas,(col//2,row//2),interpolation=cv2.INTER_CUBIC)
    print(datas.shape)
    
    datas = datas.transpose((2,1,0))
    row,col,cha = datas.shape
    datas = cv2.resize(datas,(col//2,row//2),interpolation=cv2.INTER_CUBIC)
    print(datas.shape)
    
    datas = datas.transpose((0,2,1))
    print(datas.shape)
    
    return datas

#####################################################################
#网络结构
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
edsr = EDSR().cuda()
print(edsr)
#重载入模型
if if_restore:
    print('load weight')
    edsr.load_state_dict(torch.load(os.path.join(root,'model','hcp_edsr_2_params.pkl')))
    print('load weight success')

optimizer = torch.optim.Adam(edsr.parameters(),lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
loss_function = nn.L1Loss()
#####################################################################
loss_list = []

for step in range(10000000):
    paths = glob.glob(os.path.join(root,'target_dataset','*.nrrd'))#获取训练集路径
    random.shuffle(paths)#打乱
    
    avg_loss = 0
    
    for datas_path in paths:
        original_datas,original_options = nrrd.read(datas_path)#读取nrrd数据
        original_datas = PaddingData(original_datas).astype(np.float32)#padding数据
        datas = ThreeDDown(original_datas)#下采样数据

        result = np.zeros_like(original_datas,dtype=original_datas.dtype)#构造结果矩阵
        
        #考虑到硬件处理能力的问题，对数据进行移动窗口处理
        cha_step = 20#第2维移动步数
        row_step = 20#第0维移动步数
        col_step = 20#第1维移动步数
        for cha in range(0,datas.shape[-1],cha_step):
            for row in range(0,datas.shape[0],row_step):
                for col in range(0,datas.shape[1],col_step):
                    _datas = datas[row:row+row_step,col:col+col_step,cha:cha+cha_step]#获取下采样数据的窗口数据
                    _original_datas = original_datas[row*4:(row+row_step)*4,col*4:(col+col_step)*4,cha*4:(cha+cha_step)*4]#获取下采样数据对应的真值数据，维度扩大四倍
                    
                    x = _datas.transpose((2,0,1))[np.newaxis,np.newaxis,:,:,:]#网络输入数据变换数据格式为网络需要格式
                    x =  Variable(torch.from_numpy(x)).float().cuda()#网络输入数据转换为变量并载入cuda
                    target = _original_datas.transpose((2,0,1))[np.newaxis,:,:,:]#网络真值数据变换数据格式为网络需要格式
                    target = Variable(torch.from_numpy(target)).float().cuda()#网络真值数据转换为变量并载入cuda

                    optimizer.zero_grad()#网络梯度初始化
                    out = edsr(x)#网络执行
                    loss = loss_function(out,target)#求损失
                    loss.backward()#损失反向传播
                    optimizer.step()#梯度更改，优化参数
                    
                    result[row*4:(row+row_step)*4,col*4:(col+col_step)*4,cha*4:(cha+cha_step)*4] = out.cpu().data.numpy()[0,:,:,:].transpose((1,2,0))
                    
                    avg_loss += loss.cpu().data[0]
    
    avg_loss = avg_loss/len(paths)
    print('step:{},loss:{}'.format(step,avg_loss))
    loss_list.append(avg_loss)
    
    axis = np.linspace(0, step, step+1)
    fig = plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.grid(True)
    plt.plot(axis, loss_list)
    plt.legend()
    plt.savefig('loss.pdf')
    plt.close(fig)
    
    if step%99==0:
        if step%2==0:
            torch.save(edsr,'model/hcp_edsr_2.pkl')
            torch.save(edsr.state_dict(),'model/hcp_edsr_2_params.pkl')
        else:
            torch.save(edsr,'model/hcp_edsr_1.pkl')
            torch.save(edsr.state_dict(),'model/hcp_edsr_1_params.pkl')
        
        result = (65536*((result-result.min())/(result.max()-result.min()))).astype(np.uint16)
        original_options['sizes'] = [result.shape[1],result.shape[0],result.shape[2]]
        original_options['space directions'] = [['1','0','0'],['0','1','0'],['0','0','1']]
        original_options['space origin'] = ['0','0','0']
        original_options['type'] = 'unsigned_short'
        nrrd.write('step_nrrd/step_{}.nrrd'.format(step),result,options=original_options)
