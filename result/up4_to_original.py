import numpy as np
import cv2
import nrrd

import os
import sys
import glob
import gc
gc.enable()

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

def ThreeDDown(datas,scale=4):
    
    row,col,cha = datas.shape
    new_row,new_col,new_cha = row//4,col//4,cha//4
    
    datas = cv2.resize(datas,(new_col,new_row),interpolation=cv2.INTER_CUBIC)
    
    datas = datas.transpose((1,2,0))
    datas = cv2.resize(datas,(new_cha,new_col),interpolation=cv2.INTER_CUBIC)
    
    datas = datas.transpose((2,0,1))
    
    return datas

if __name__=='__main__':
    input_name = sys.argv[1]
    output_name = sys.argv[2]

    cha_step = 40
    row_step = 40
    col_step = 40
    offect = 4

    #datas_path = os.path.join(root,input_name)
    datas_path = input_name
    original_datas,original_options = nrrd.read(datas_path)
    original_shape = original_datas.shape

    datas = original_datas.astype(np.float32)
    datas = PaddingData(datas,scale=40)

    one = np.zeros((datas.shape[0],datas.shape[1],offect))
    _padding_datas = np.concatenate((one,datas,one),axis=2)
    two = np.zeros((offect,_padding_datas.shape[1],_padding_datas.shape[2]))
    _padding_datas = np.concatenate((two,_padding_datas,two),axis=0)
    three = np.zeros((_padding_datas.shape[0],offect,_padding_datas.shape[2]))
    _padding_datas = np.concatenate((three,_padding_datas,three),axis=1)

    padding_shape = list(map(lambda x:x[0]-x[1],zip(_padding_datas.shape,original_shape)))

    result = np.zeros((_padding_datas.shape[0]//4,_padding_datas.shape[1]//4,_padding_datas.shape[2]//4),dtype=datas.dtype)

    for cha in range(offect,datas.shape[-1]+offect,cha_step):
        for row in range(offect,datas.shape[0]+offect,row_step):
            for col in range(offect,datas.shape[1]+offect,col_step):
                _datas = _padding_datas[row-offect:row+row_step+offect,col-offect:col+col_step+offect,cha-offect:cha+cha_step+offect]
                
                _datas = ThreeDDown(_datas)
                
                result[(row-offect)//4:(row+row_step-offect)//4,(col-offect)//4:(col+col_step-offect)//4,(cha-offect)//4:(cha+cha_step-offect)//4] = _datas[offect//4:-offect//4,offect//4:-offect//4,offect//4:-offect//4]

    result = result[:-padding_shape[0]//4,:-padding_shape[1]//4,:-padding_shape[2]//4]

    result = (65535*((result-result.min())/(result.max()-result.min()))).astype(np.uint16)

    original_options['sizes'] = [result.shape[1],result.shape[0],result.shape[2]]
    original_options['space directions'] = [['1','0','0'],['0','1','0'],['0','0','1']]
    original_options['space origin'] = ['0','0','0']
    original_options['type'] = 'unsigned_short'
    nrrd.write(output_name,result,options=original_options)