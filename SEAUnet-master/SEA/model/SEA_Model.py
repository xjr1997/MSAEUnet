import SEA.segmentation_models as sm
from keras_radam import RAdam
import tensorflow as tf
import os
from keras import Model, layers, optimizers
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
import keras
import keras.backend as K
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda,Conv1D
from keras.optimizers import *
from keras.models import Model
from torch import nn

from keras.layers import Input,Conv2DTranspose, MaxPooling2D,BatchNormalization,concatenate,Activation


def Dias_SEA(cfgs):
    """
    Build deep learning model from configuration file
    """
    backbone = cfgs['Model']['Backbone']
    if backbone == 'None':
        backbone = None
    backbone_weights = cfgs['Model']['BackboneWeights']

    if backbone_weights == 'None':
        backbone_weights = None

    class_num = int(cfgs['Data']['ClassNum'])

    learning_rate = float(cfgs['Model']['LearningRate'])

    optimizer =  cfgs['Model']['Optimizer']
    if optimizer == 'RAdam':
        optimizer = RAdam(learning_rate=learning_rate)
    elif optimizer == 'Adam':
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'SGD':
        optimizer = optimizers.SGD(learning_rate=learning_rate)
    
    loss = cfgs['Model']['Loss']
    if loss == 'bce':
        loss = sm.losses.binary_crossentropy
    elif loss == 'dice':
        loss = sm.losses.dice_loss
    elif loss == 'BinaryFocalLoss':
        loss = sm.losses.binary_focal_loss
    
    metric = cfgs['Model']['Metric']
    if metric == 'F_score':
        metric = sm.metrics.FScore()
    elif metric == 'IoU':
        metric = sm.metrics.IOUScore()
    
        
    model = sm.SEA(backbone_name=backbone, 
                    encoder_weights=backbone_weights,
                    classes=class_num)

    
    model.compile(
        optimizer,
        loss = loss,
        metrics=[metric]
    )
    model.summary()
    return model







