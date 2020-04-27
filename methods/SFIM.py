# -*- coding: utf-8 -*-
"""
License: Apache-2.0
@author: gaj
E-mail: anjing_guo@hnu.edu.cn
Paper References:
    [1] J. Liu, “Smoothing filter based intensity modulation: a spectral preserve image fusion technique for improving spatial details,”
        International Journal of Remote Sensing, vol. 21, no. 18, pp. 3461–3472, December 2000.
    [2] G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
        IEEE Transaction on Geoscience and Remote Sensing, 2014.
"""

import numpy as np
from utils import upsample_interp23
from scipy import signal

def SFIM(pan, hs):

    M, N, c = pan.shape
    m, n, C = hs.shape
    
    ratio = int(np.round(M/m))
        
    print('get sharpening ratio: ', ratio)
    assert int(np.round(M/m)) == int(np.round(N/n))
    
    #upsample
    u_hs = upsample_interp23(hs, ratio)
    
    if np.mod(ratio, 2)==0:
        ratio = ratio + 1
        
    pan = np.tile(pan, (1, 1, C))
    
    pan = (pan - np.mean(pan, axis=(0, 1)))*(np.std(u_hs, axis=(0, 1), ddof=1)/np.std(pan, axis=(0, 1), ddof=1))+np.mean(u_hs, axis=(0, 1))
    
    kernel = np.ones((ratio, ratio))
    kernel = kernel/np.sum(kernel)
    
    I_SFIM = np.zeros((M, N, C))
    for i in range(C):
        lrpan = signal.convolve2d(pan[:, :, i], kernel, mode='same', boundary = 'wrap')
        I_SFIM[:, :, i] = u_hs[:, :, i]*pan[:, :, i]/(lrpan+1e-8)

    #adjustment
    I_SFIM[I_SFIM<0]=0
    I_SFIM[I_SFIM>1]=1    
    
    return np.uint8(I_SFIM*255)