import tensorflow as tf
import numpy as np
from SEA.model.FPN_Model import SEA_FPN
from SEA.model.Unet_Model import SEA_Unet
from SEA.model.SEA_Model import SEA_SEA
from SEA.dataIO.data import IonoDataManager
import segmentation_models as sm
import matplotlib.pyplot as plt
import os

def train(cfgs):
    """
    Function for training SEA.
    """
    print('Setting Model...')
    if cfgs['Model']['Type'] == 'Unet' or cfgs['Model']['Type'] == 'naiveUnet':
        model = SEA_Unet(cfgs)
        
    elif cfgs['Model']['Type'] == 'FPN':
        model = SEA_FPN(cfgs)
    elif cfgs['Model']['Type'] == 'SEA':
        model = SEA_SEA(cfgs)
       
    dataManager = IonoDataManager(cfgs)
    # check data IO
    total_step = int(cfgs['Train']['TotalStep'])
    # print interval
    print_interval = int(cfgs['Train']['PrintInterval'])
    # plot interval
    plot_inverval = int(cfgs['Train']['PlotInterval'])
    # save interval
    save_inverval = int(cfgs['Train']['SaveInterval'])
    # plot image save directory
    img_save_dir = cfgs['Train']['ImgSaveDir']
    if os.path.exists(img_save_dir):
        pass
    else:
        os.makedirs(img_save_dir)
    # history log directory
    hist_log_dir = cfgs['Train']['HistLogDir']
    if os.path.exists(hist_log_dir):
        pass
    else:
        os.makedirs(hist_log_dir)
    # model save directory
    model_save_dir = cfgs['Train']['ModelSaveDir']
    if os.path.exists(model_save_dir):
        pass
    else:
        os.makedirs(model_save_dir)
    hist_list = list()
    print('Start Training')
    # start training
    for idx in range(total_step):
        x_train, y_train = dataManager.get_train_batch()
        hist = model.train_on_batch(x=x_train,y=y_train)
        hist_list.append(hist)
        
        # Print history information
        if idx % print_interval == 0:
            print('On Step {}'.format(idx))
            print(hist)

        # Save image result
        if idx % plot_inverval == 0:
            x_train, y_train = dataManager.get_train_batch()
            y_test = model.predict(x_train)
            plt.figure(figsize=(24,8))
            plt.subplot(1,3,1)
            plt.imshow(x_train[0,:,:,:],origin='lower')
            plt.subplot(1,3,2)
            plt.imshow(y_train[0,:,:,:],origin='lower')
            plt.subplot(1,3,3)
            plt.imshow(y_test[0,:,:,:],origin='lower')
            plt.savefig(img_save_dir+'STEP_{}.png'.format(idx),dpi=300)
            plt.close()

        # Save model
        if idx % save_inverval == 0:
            model.save(model_save_dir+'STEP_{}.model'.format(idx))
            np.save(model_save_dir+'hist.npy',hist_list)
    
    return
        
