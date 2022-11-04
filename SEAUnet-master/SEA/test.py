import tensorflow as tf
import numpy as np
from SEA.model.FPN_Model import SEA_FPN
from SEA.model.Unet_Model import SEA_Unet
from SEA.model.Unet_Model import SEA_SEA
from SEA.dataIO.data import IonoDataManager
from SEA.dataIO.dataPostProcess import get_minH_maxF
import segmentation_models as sm
import matplotlib.pyplot as plt
import os

def test(cfgs):
    print('Setting Model...')
    # set up model
    if cfgs['Model']['Type'] == 'Unet' or cfgs['Model']['Type'] == 'naiveUnet':
        model = SEA_Unet(cfgs)
    elif cfgs['Model']['Type'] == 'FPN':
        model = SEA_FPN(cfgs)
    elif cfgs['Model']['Type'] == 'SEA':
        model = SEA_SEA(cfgs)
    model.load_weights(cfgs['Test']['ModelPath'])
    
    threshold = float(cfgs['Test']['Threshold'])
    img_save_dir = cfgs['Test']['ImgSaveDir']
    # set up dataset
    dataManager = IonoDataManager(cfgs)
    all_test_num = len(dataManager.test_data_list)

    # if only save MinH and MaxF
    if cfgs['Test']['TestSave'] == 'OnlyMinHMaxF':
        res_mat = np.zeros([len(dataManager.test_data_list),3,6])
        for idx in range(len(dataManager.test_data_list)):
            test_data, human_res, artist_res = dataManager.get_test_batch(idx)
            SEA_res = model.predict(test_data)
            res_mat[idx,:,0:2] = get_minH_maxF(human_res,threshold)
            res_mat[idx,:,2:4] = get_minH_maxF(artist_res,threshold)
            res_mat[idx,:,4:6] = get_minH_maxF(SEA_res,threshold)
            
            if idx%10==0:
                plt.figure(figsize=(24,8))
                plt.subplot(1,4,4)
                plt.imshow(test_data[0,:,:,:],origin='lower')
                plt.subplot(1,4,3)
                plt.imshow(human_res[0,:,:,:],origin='lower')
                plt.subplot(1,4,2)
                plt.imshow(artist_res[0,:,:,:],origin='lower')
                plt.subplot(1,4,1)
                plt.imshow(SEA_res[0,:,:,:],origin='lower')
                plt.savefig(img_save_dir+'STEP_{}.png'.format(idx),dpi=300)
                plt.close()
                print(res_mat[idx,:])
        np.save(cfgs['Test']['SavePath']+'MinHMaxF.npy',res_mat)
        
    # if save All outputs of the model
    elif cfgs['Test']['TestSave'] == 'AllOutput':
        for idx in range(all_test_num):
            print('On {}'.format(idx))
            if cfgs['Test']['ScaleOnly']:
                test_data = dataManager.get_scale_only(idx)
                SEA_res = model.predict(test_data)
                res_mat = (test_data,SEA_res)
                np.save(cfgs['Test']['SavePath']+'TEST_{}.npy'.format(idx),res_mat)
            else:
                test_data, human_res, artist_res = dataManager.get_test_batch(idx)
                SEA_res = model.predict(test_data)
                res_mat = (test_data, human_res, artist_res,SEA_res)
                np.save(cfgs['Test']['SavePath']+'TEST_{}.npy'.format(idx),res_mat)
    else:
        print('Please choose a vaild TestSave Option')
    
    return
