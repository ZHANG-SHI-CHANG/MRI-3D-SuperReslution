import numpy as np
import nrrd

import sys

if __name__=='__main__':
    data_names = sys.argv[1]
    result_names = sys.argv[2]

    data_list = data_names.split(',')
    result_list = result_names.split(',')
    assert len(data_list)==len(result_list)
    print('all data:{}'.format(data_list))

    for i,data_path in enumerate(data_list):
        data,option = nrrd.read(data_path)
        data[:,:,:] = data[::-1,:,:]
        data[:,:,:] = data[:,::-1,:]
        nrrd.write(result_list[i],data,option)
        print('transpose data：{} to result:{}'.format(data_path,result_list[i]))
