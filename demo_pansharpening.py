# -*- coding: utf-8 -*-
"""
License: MIT
@author: gaj
E-mail: anjing_guo@hnu.edu.cn
"""

import numpy as np
import cv2
import os
import scipy.io as sio

from methods.Bicubic import Bicubic
from methods.Brovey import Brovey
from methods.PCA import PCA
from methods.IHS import IHS
from methods.SFIM import SFIM
from methods.GS import GS
from methods.Wavelet import Wavelet
from methods.MTF_GLP import MTF_GLP
from methods.MTF_GLP_HPM import MTF_GLP_HPM
from methods.GSA import GSA
from methods.CNMF import CNMF
from methods.GFPCA import GFPCA
from methods.PNN import PNN
from methods.PanNet import PanNet

#'''loading data'''
#used_ms = np.load('./images/GF2_BJ_mss.npy')
#used_pan = np.load('./images/GF2_BJ_pan.npy')
#used_pan = np.expand_dims(used_pan, -1)

data = sio.loadmat('./images/imgWV2.mat')
used_ms = data['I_MS']
used_pan = data['I_PAN']
used_pan = np.expand_dims(used_pan, -1)

'''normalization'''
max_patch, min_patch = np.max(used_ms, axis=(0,1)), np.min(used_ms, axis=(0,1))
used_ms = np.float32(used_ms-min_patch) / (max_patch - min_patch)
max_patch, min_patch = np.max(used_pan, axis=(0,1)), np.min(used_pan, axis=(0,1))
used_pan = np.float32(used_pan-min_patch) / (max_patch - min_patch)

print('ms shape: ', used_ms.shape, 'pan shape: ', used_pan.shape)

save_dir='./results/'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

'''here is the main function'''
fused_image = GSA(used_pan[:, :, :], used_ms[:, :, :])

cv2.imwrite(save_dir+'GSA.tiff', fused_image[:, :, [2, 3, 5]])