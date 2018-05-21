import numpy as np
import cv2
import nrrd

import random
import os
import glob

root = os.getcwd()

for nrrd_data_path in glob.glob(os.path.join(root,'original_dataset','*.nrrd')):
    original_datas,original_options = nrrd.read(nrrd_data_path)
    
    nrrd_name = nrrd_data_path.split('\\')[-1][:-5]
    
    count = 0
    save_count = 0
    
    for cha in range(0,original_datas.shape[-1]-80,40):
        for row in range(0,original_datas.shape[0]-80,40):
            for col in range(0,original_datas.shape[1]-80,40):
                
                _original_datas = original_datas[row:row+80,col:col+80,cha:cha+80]
                _original_datas = (65536*((_original_datas-_original_datas.min())/(_original_datas.max()-_original_datas.min()))).astype(np.uint16)
                print('{} process shape {}'.format(nrrd_name,_original_datas.shape))
                
                if np.where((_original_datas>=20))[0].shape[0] >= (80*80*80/2):
                    original_options['sizes'] = [_original_datas.shape[1],_original_datas.shape[0],_original_datas.shape[2]]
                    original_options['space directions'] = [['1','0','0'],['0','1','0'],['0','0','1']]
                    original_options['space origin'] = ['0','0','0']
                    original_options['type'] = 'unsigned_short'
                    save_path = os.path.join(root,'target_dataset','{}_{}.nrrd'.format(nrrd_name,count))
                    
                    random_save = random.randint(6,20)
                    if count%random_save==0:
                        if save_count<15:
                            nrrd.write(save_path,_original_datas,options=original_options)
                            save_count += 1
                        else:
                            pass
                    else:
                        pass
                    
                    count += 1
                else:
                    print('abandon {} shape {}'.format(nrrd_name,_original_datas.shape))