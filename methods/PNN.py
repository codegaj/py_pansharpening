# -*- coding: utf-8 -*-
"""
License: MIT
@author: gaj
E-mail: anjing_guo@hnu.edu.cn
Code Reference: https://github.com/sergiovitale/pansharpening-cnn-python-version
Paper References:
    Masi G, Cozzolino D, Verdoliva L, et al. Pansharpening by convolutional neural networks
    [J]. Remote Sensing, 2016, 8(7): 594.
"""

import numpy as np
from keras.layers import Concatenate, Conv2D, Input
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
import tensorflow as tf
from tqdm import tqdm
from keras import backend as K
import os
import random
from utils import upsample_interp23, downgrade_images
import gc

def psnr(y_true, y_pred):
    """Peak signal-to-noise ratio averaged over samples and channels."""
    mse = K.mean(K.square(y_true*255 - y_pred*255), axis=(-3, -2, -1))
    return K.mean(20 * K.log(255 / K.sqrt(mse)) / np.log(10))

def pnn_net(lrhs_size=(32, 32, 3), hrms_size = (32, 32, 1)):
    
    lrhs_inputs = Input(lrhs_size)
    hrms_inputs = Input(hrms_size)
    
    mixed = Concatenate()([lrhs_inputs, hrms_inputs])

    mixed1 = Conv2D(64, (9, 9), strides=(1, 1), padding='same', activation='relu')(mixed)

    mixed1 = Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation='relu')(mixed1)
    
    c6 = Conv2D(lrhs_size[2], (5, 5), strides=(1, 1), padding='same', activation='relu', name='model1_last1')(mixed1)
    
    model = Model(inputs = [lrhs_inputs, hrms_inputs], outputs = c6)

    model.compile(optimizer =Adam(lr = 5e-4), loss = 'mse', metrics=[psnr])
    
    model.summary()

    return model

def PNN(hrms, lrhs, sensor = None):
    """
    this is an zero-shot learning method with deep learning (PNN)
    hrms: numpy array with MXNXc
    lrhs: numpy array with mxnxC
    """
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    
    M, N, c = hrms.shape
    m, n, C = lrhs.shape
    
    stride = 8
    training_size=32#training patch size
    testing_size=400#testing patch size
    reconstructing_size=320#reconstructing patch size
    left_pad = (testing_size-reconstructing_size)//2
    

    '''
        testing
    ---------------
    |     rec     |
    |   -------   |
    |   |     |   |
    |   |     |   |
    |   -------   |
    |             |
    ---------------
    |pad|
    
    '''
    
    ratio = int(np.round(M/m))
        
    print('get sharpening ratio: ', ratio)
    assert int(np.round(M/m)) == int(np.round(N/n))
    
    train_hrhs_all = []
    train_hrms_all = []
    train_lrhs_all = []
    
    used_hrhs = lrhs
    used_lrhs = lrhs
    
    used_lrhs, used_hrms = downgrade_images(used_lrhs, hrms, ratio, sensor=sensor)
    
    print(used_lrhs.shape, used_hrms.shape)
    
    used_lrhs = upsample_interp23(used_lrhs, ratio)
    
    """crop images"""
    print('croping images...')
    
    for j in range(0, used_hrms.shape[0]-training_size, stride):
        for k in range(0, used_hrms.shape[1]-training_size, stride):
            
            temp_hrhs = used_hrhs[j:j+training_size, k:k+training_size, :]
            temp_hrms = used_hrms[j:j+training_size, k:k+training_size, :]
            temp_lrhs = used_lrhs[j:j+training_size, k:k+training_size, :]
            
            train_hrhs_all.append(temp_hrhs)
            train_hrms_all.append(temp_hrms)
            train_lrhs_all.append(temp_lrhs)
            
    train_hrhs_all = np.array(train_hrhs_all, dtype='float16')
    train_hrms_all = np.array(train_hrms_all, dtype='float16')
    train_lrhs_all = np.array(train_lrhs_all, dtype='float16')
    
    index = [i for i in range(train_hrhs_all.shape[0])]
#    random.seed(2020)
    random.shuffle(index)
    train_hrhs = train_hrhs_all[index, :, :, :]
    train_hrms= train_hrms_all[index, :, :, :]
    train_lrhs = train_lrhs_all[index, :, :, :]
    
    print(train_hrhs.shape, train_hrms.shape, train_lrhs.shape)
    
    """train net"""
    print('training...')
    
    def lr_schedule(epoch):
        """Learning Rate Schedule
    
        # Arguments
            epoch (int): The number of epochs
    
        # Returns
            lr (float32): learning rate
        """
        lr = 5e-4
        if epoch > 40:
            lr *= 1e-2
        elif epoch > 20:
            lr *= 1e-1
        return lr
    
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    checkpoint = ModelCheckpoint(filepath='./weights/PNN_model.h5',
                             monitor='val_psnr',
                             mode='max',
                             verbose=1,
                             save_best_only=True)
    callbacks = [lr_scheduler, checkpoint]
    
    model = pnn_net(lrhs_size=(training_size, training_size, C), hrms_size=(training_size, training_size, c))
        
    model.fit( x=[train_lrhs, train_hrms],
                y=train_hrhs,
                validation_split=0.33,
                batch_size=32,
                epochs=50,
                verbose=1,
                callbacks=callbacks)
    
    model = pnn_net(lrhs_size=(testing_size, testing_size, C), hrms_size=(testing_size, testing_size, c))
    
    model.load_weights('./weights/PNN_model.h5')
    
    """eval"""
    print('evaling...')
    
    new_M = min(M, m*ratio)
    new_N = min(N, n*ratio)
    
    print('output image size:', new_M, new_N)
    
    test_label = np.zeros((new_M, new_N, C), dtype = 'uint8')
    
    used_lrhs = lrhs[:new_M//ratio, :new_N//ratio, :]
    used_hrms = hrms[:new_M, :new_N, :]
    
    used_lrhs = upsample_interp23(used_lrhs, ratio)
    
    used_lrhs = np.expand_dims(used_lrhs, 0)
    used_hrms = np.expand_dims(used_hrms, 0)
    
    used_lrhs = np.pad(used_lrhs, ((0, 0), (left_pad, testing_size), (left_pad, testing_size), (0, 0)), mode='symmetric')
    used_hrms = np.pad(used_hrms, ((0, 0), (left_pad, testing_size), (left_pad, testing_size), (0, 0)), mode='symmetric')
    
    for h in tqdm(range(0, new_M, reconstructing_size)):
        for w in range(0, new_N, reconstructing_size):
            temp_lrhs = used_lrhs[:, h:h+testing_size, w:w+testing_size, :]
            temp_hrms = used_hrms[:, h:h+testing_size, w:w+testing_size, :]
            
            fake = model.predict([temp_lrhs, temp_hrms])
            fake = np.clip(fake, 0, 1)
            fake.shape=(testing_size, testing_size, C)
            fake = fake[left_pad:(testing_size-left_pad), left_pad:(testing_size-left_pad)]
            fake = np.uint8(fake*255)
            
            if h+testing_size>new_M:
                fake = fake[:new_M-h, :, :]
                
            if w+testing_size>new_N:
                fake = fake[:, :new_N-w, :]
            
            test_label[h:h+reconstructing_size, w:w+reconstructing_size]=fake
    
    K.clear_session()
    gc.collect()
    del model
    
    return np.uint8(test_label)