# -*- coding: utf-8 -*-
"""
License: MIT
@author: gaj
E-mail: anjing_guo@hnu.edu.cn
Paper References:
    [1] W. Liao et al., "Two-stage fusion of thermal hyperspectral and visible RGB image by PCA and guided filter," 
        2015 7th Workshop on Hyperspectral Image and Signal Processing: Evolution in Remote Sensing (WHISPERS), Tokyo, 2015, pp. 1-4.
"""

import numpy as np
from utils import upsample_interp23
from sklearn.decomposition import PCA as princomp
from cv2.ximgproc import guidedFilter

def GFPCA(pan, hs):

    M, N, c = pan.shape
    m, n, C = hs.shape
    
    ratio = int(np.round(M/m))
        
    print('get sharpening ratio: ', ratio)
    assert int(np.round(M/m)) == int(np.round(N/n))
    
    p = princomp(n_components=C)
    pca_hs = p.fit_transform(np.reshape(hs, (m*n, C)))
    
    pca_hs = np.reshape(pca_hs, (m, n, C))
    
    pca_hs = upsample_interp23(pca_hs, ratio)
    
    gp_hs = []
    for i in range(C):
        temp = guidedFilter(np.float32(pan), np.float32(np.expand_dims(pca_hs[:, :, i], -1)), 8, eps = 0.001**2)
        temp = np.expand_dims(temp ,axis=-1)
        gp_hs.append(temp)
        
    gp_hs = np.concatenate(gp_hs, axis=-1)
    
    I_GFPCA = p.inverse_transform(gp_hs)
    
    #adjustment
    I_GFPCA[I_GFPCA<0]=0
    I_GFPCA[I_GFPCA>1]=1
    
    return np.uint8(I_GFPCA*255)