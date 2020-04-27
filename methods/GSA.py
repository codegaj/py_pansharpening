# -*- coding: utf-8 -*-
"""
License: MIT
@author: gaj
E-mail: anjing_guo@hnu.edu.cn
Paper References:
    [1] B. Aiazzi, S. Baronti, and M. Selva, “Improving component substitution Pansharpening through multivariate regression of MS+Pan data,” 
        IEEE Transactions on Geoscience and Remote Sensing, vol. 45, no. 10, pp. 3230–3239, October 2007.
    [2] G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
        IEEE Transaction on Geoscience and Remote Sensing, 2014.
"""

import numpy as np
from utils import upsample_interp23
import cv2

def estimation_alpha(pan, hs, mode='global'):
    if mode == 'global':
        IHC = np.reshape(pan, (-1, 1))
        ILRC = np.reshape(hs, (hs.shape[0]*hs.shape[1], hs.shape[2]))
        
        alpha = np.linalg.lstsq(ILRC, IHC)[0]
        
    elif mode == 'local':
        patch_size = 32
        all_alpha = []
        print(pan.shape)
        for i in range(0, hs.shape[0]-patch_size, patch_size):
            for j in range(0, hs.shape[1]-patch_size, patch_size):
                patch_pan = pan[i:i+patch_size, j:j+patch_size, :]
                patch_hs = hs[i:i+patch_size, j:j+patch_size, :]
                
                IHC = np.reshape(patch_pan, (-1, 1))
                ILRC = np.reshape(patch_hs, (-1, hs.shape[2]))
                
                local_alpha = np.linalg.lstsq(ILRC, IHC)[0]
                all_alpha.append(local_alpha)
                
        all_alpha = np.array(all_alpha)
        
        alpha = np.mean(all_alpha, axis=0, keepdims=False)
        
    return alpha

def GSA(pan, hs):
    
    M, N, c = pan.shape
    m, n, C = hs.shape
    
    ratio = int(np.round(M/m))
        
    print('get sharpening ratio: ', ratio)
    assert int(np.round(M/m)) == int(np.round(N/n))
    
    #upsample
    u_hs = upsample_interp23(hs, ratio)
    
    #remove means from u_hs
    means = np.mean(u_hs, axis=(0, 1))
    image_lr = u_hs-means
    
    #remove means from hs
    image_lr_lp = hs-np.mean(hs, axis=(0,1))
    
    #sintetic intensity
    image_hr = pan-np.mean(pan)
    image_hr0 = cv2.resize(image_hr, (n, m), cv2.INTER_CUBIC)
    image_hr0 = np.expand_dims(image_hr0, -1)
    
    alpha = estimation_alpha(image_hr0, np.concatenate((image_lr_lp, np.ones((m, n, 1))), axis=-1), mode='global')
    
    I = np.dot(np.concatenate((image_lr, np.ones((M, N, 1))), axis=-1), alpha)
    
    I0 = I-np.mean(I)
    
    #computing coefficients
    g = []
    g.append(1)
    
    for i in range(C):
        temp_h = image_lr[:, :, i]
        c = np.cov(np.reshape(I0, (-1,)), np.reshape(temp_h, (-1,)), ddof=1)
        g.append(c[0,1]/np.var(I0))
    g = np.array(g)
    
    #detail extraction
    delta = image_hr-I0
    deltam = np.tile(delta, (1, 1, C+1))
    
    #fusion
    V = np.concatenate((I0, image_lr), axis=-1)
    
    g = np.expand_dims(g, 0)
    g = np.expand_dims(g, 0)
    
    g = np.tile(g, (M, N, 1))
    
    V_hat = V + g*deltam
    
    I_GSA = V_hat[:, :, 1:]
    
    I_GSA = I_GSA - np.mean(I_GSA, axis=(0, 1)) + means
    
    #adjustment
    I_GSA[I_GSA<0]=0
    I_GSA[I_GSA>1]=1
    
    return np.uint8(I_GSA*255)