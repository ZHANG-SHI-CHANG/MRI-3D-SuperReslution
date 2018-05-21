import numpy as np
import cv2
import nrrd#pip install pynrrd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import gc
gc.enable()

import os
import sys
import glob

label_list = [[42,132,154,128,182,194,146,176,190,142,198,196,178,200,202,124,160,126,130,134,138,140,144,148,150,152,154,156,158,162,164,166,168,170,172,174,184,180,113],
              [43,195,199,191,177,203,143,123,129,107,205,145,165,133,155,201,187,189,193,197,207,183,185,181,179,175,173,171,169,167,163,161,159,157,153,151,149,147,141,135,139],
              [38],
              [39],
              [8,44,67,111,64,97,69],
              [45,102,106,76,112,108,110,114,116,122,66,136,51,53,188,192,206,61],
              [40],
              [41,103,119,105,101,5],
              [31],
              [32],
              [35,71,73,72],
              [23,36],
              [30],
              [47],
              [48],
              [4,11,15],
              [52],
              [55],
              [56],
              [57],
              [58],
              [59],
              [60],
              [61],
              [62]
              ]

if __name__=='__main__':
    input_name = sys.argv[1]
    
    root = os.getcwd()

    for nrrd_data_path in glob.glob(os.path.join(root,input_name)):
        original_datas,original_options = nrrd.read(nrrd_data_path)
        
        nrrd_name = nrrd_data_path.split('/')[-1][:-5]
        
        save_root_path = os.path.join(root,'result_pngs',nrrd_name)
        if not os.path.exists(save_root_path):
            os.makedirs(save_root_path)
        
        for i in range(3):
            save_path = os.path.join(save_root_path,str(i))
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            for png_count in range(original_datas.shape[i]):
                if i==0:
                    png = original_datas[png_count,:,:]
                elif i==1:
                    png = original_datas[:,png_count,:]
                else:
                    png = original_datas[:,:,png_count]
                
                png = png.T[::-1,:].astype(np.float32)
                '''
                png = (png-png[0,0])/(png.max()-png[0,0])
                png[png<0] = 0
                png = (256*png).astype(np.uint8)
                '''
                png = (png).astype(np.uint8)
                one_png = np.ones_like(png)
                
                for i,_label_list in enumerate(label_list):
                    for label in _label_list:
                        png[png==label*one_png]=i
                
                #cv2.imwrite(os.path.join(save_path,'{}.png'.format(png_count)),png)
                print('save {}.png from {} of {}'.format(png_count,nrrd_name,i))
                
                fig  = plt.figure()
                ax = fig.add_subplot(1,1,1)
                gci = ax.imshow((png).astype(np.uint8),cmap='jet')
                cbar = plt.colorbar(gci)
                cbar.set_ticks(np.linspace(0,24,25))
                plt.savefig(os.path.join(save_path,'{}.png'.format(png_count)),bbox_inches='tight',dpi=200)
                
                plt.cla()
                plt.clf()
                plt.close('all')
                gc.collect()