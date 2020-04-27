# -*- coding: utf-8 -*-
"""
License: MIT
@author: gaj
E-mail: anjing_guo@hnu.edu.cn
Code Reference: https://github.com/oyam/PanNet-Landsat
Paper References:
    [1] Yang J, Fu X, Hu Y, et al. PanNet: A deep network architecture for pan-sharpening
        [C]//Proceedings of the IEEE International Conference on Computer Vision. 2017: 5449-5457.
Notice: I remove the BN layer in the ResBlovck acquiescently, you can use it by removing the uncomment.
"""

import numpy as np
from keras.layers import Concatenate, Conv2D, Input, Layer, Add, Activation, BatchNormalization
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
import tensorflow as tf
from tqdm import tqdm
from keras import backend as K
import os
import random
from utils import downgrade_images
import gc

def psnr(y_true, y_pred):
    """Peak signal-to-noise ratio averaged over samples and channels."""
    mse = K.mean(K.square(y_true*255 - y_pred*255), axis=(-3, -2, -1))
    return K.mean(20 * K.log(255 / K.sqrt(mse)) / np.log(10))

class hp_filter(Layer):
    def __init__(self, **kwargs):
        super(hp_filter, self).__init__(**kwargs)

    def call(self, inputs):
        
        c = inputs.get_shape().as_list()[-1]
        
        kernel = np.ones((5,5))/25.0
        kernel = K.constant(kernel)
        
        kernel = K.expand_dims(kernel, -1)
        kernel = K.expand_dims(kernel, -1)
        
        kernel = K.tile(kernel, (1, 1, c, 1))
        
        outs = K.depthwise_conv2d(inputs, kernel, strides=(1, 1), padding='same')
        
        outs = inputs - outs
        
        self.outs_size = outs.get_shape().as_list()
        
        return outs

    def compute_output_shape(self, input_shape):
        return tuple(self.outs_size)
    
    def get_config(self):
        config = super(hp_filter, self).get_config()
        return config

class resize(Layer):
    def __init__(self, target_size,
                 **kwargs):
        self.target_size = (target_size[0], target_size[1])
        super(resize, self).__init__(**kwargs)

    def call(self, inputs):
        temp = tf.image.resize_bicubic(inputs, self.target_size)
        return temp

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.target_size[0], self.target_size[1], input_shape[3])
    
    def get_config(self):
        config = super(resize, self).get_config()
        return config
    
def conv_block(inputs, block_name='1'):
    
    conv1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name=block_name+'_1')(inputs)
#    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name=block_name+'_2')(conv1)
#    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    outputs = Add()([inputs, conv2])
    return outputs

def pannet(lrhs_size=(16, 16, 3), hrms_size = (64, 64, 1)):
    
    lrhs_inputs = Input(lrhs_size)
    hrms_inputs = Input(hrms_size)
    
    h_lrhs = hp_filter()(lrhs_inputs)
    h_hrms = hp_filter()(hrms_inputs)
    
    re_h_lrhs = resize(hrms_size)(h_lrhs)
    re_lrhs = resize(hrms_size)(lrhs_inputs)
    
    mixed = Concatenate()([re_h_lrhs, h_hrms])

    mixed1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(mixed)
    
    x = mixed1
    for i in range(4):
        x = conv_block(x, str(i))
    
    x = Conv2D(lrhs_size[2], (3, 3), strides=(1, 1), padding='same', name='model1_last1')(x)
    
    last = Add()([x, re_lrhs])
    
    model = Model(inputs = [lrhs_inputs, hrms_inputs], outputs = last)

    model.compile(optimizer=Adam(lr = 5e-4), loss = 'mae', metrics=[psnr])
    
    model.summary()

    return model

def PanNet(hrms, lrhs, sensor = None):
    """
    this is an zero-shot learning method with deep learning (PanNet)
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
    training_size=64#training patch size
    testing_size=400#testing patch size
    reconstructing_size=320#reconstructing patch size to avoid boundary effect
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
    
    """crop images"""
    print('croping images...')
    
    for j in range(0, used_hrms.shape[0]-training_size, stride):
        for k in range(0, used_hrms.shape[1]-training_size, stride):
            
            temp_hrhs = used_hrhs[j:j+training_size, k:k+training_size, :]
            temp_hrms = used_hrms[j:j+training_size, k:k+training_size, :]
            temp_lrhs = used_lrhs[int(j/4):int((j+training_size)/4), int(k/4):int((k+training_size)/4), :]
            
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
    checkpoint = ModelCheckpoint(filepath='./weights/PANNET_model.h5',
                             monitor='val_psnr',
                             mode='max',
                             verbose=1,
                             save_best_only=True)
    callbacks = [lr_scheduler, checkpoint]
    
    model = pannet(lrhs_size=(int(training_size/ratio), int(training_size/ratio), C), hrms_size=(training_size, training_size, c))
        
    model.fit( x=[train_lrhs, train_hrms],
                y=train_hrhs,
                validation_split=0.1,
                batch_size=32,
                epochs=50,
                verbose=1,
                callbacks=callbacks)
    
    model = pannet(lrhs_size=(int(testing_size/ratio), int(testing_size/ratio), C), hrms_size=(testing_size, testing_size, c))
    
    model.load_weights('./weights/PANNET_model.h5')
    
    """eval"""
    print('evaling...')
        
    used_lrhs = np.expand_dims(lrhs, 0)
    used_hrms = np.expand_dims(hrms, 0)
    
    new_M = min(M, m*ratio)
    new_N = min(N, n*ratio)
    
    print('output image size:', new_M, new_N)
    
    test_label = np.zeros((new_M, new_N, C), dtype = 'uint8')
    
    used_lrhs = used_lrhs[:, :new_M//ratio, :new_N//ratio, :]
    used_hrms = used_hrms[:, :new_M, :new_N, :]
    
    used_lrhs = np.pad(used_lrhs, ((0, 0), (left_pad//ratio, testing_size//ratio), (left_pad//ratio, testing_size//ratio), (0, 0)), mode='symmetric')
    used_hrms = np.pad(used_hrms, ((0, 0), (left_pad, testing_size), (left_pad, testing_size), (0, 0)), mode='symmetric')
    
    for h in tqdm(range(0, new_M, reconstructing_size)):
        for w in range(0, new_N, reconstructing_size):
            temp_lrhs = used_lrhs[:,int(h/ratio):int((h+testing_size)/ratio), int(w/ratio):int((w+testing_size)/ratio), :]
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
            
#    K.clear_session()
#    gc.collect()
#    del model
    
    return np.uint8(test_label)