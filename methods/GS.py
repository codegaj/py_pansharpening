# -*- coding: utf-8 -*-
"""
License: MIT
@author: gaj
E-mail: anjing_guo@hnu.edu.cn
Paper References:
    [1] C. A. Laben and B. V. Brower, “Process for enhancing the spatial resolution of multispectral imagery using pan-sharpening,” 
        Eastman Kodak Company, Tech. Rep. US Patent # 6,011,875, 2000.
    [2] B. Aiazzi, S. Baronti, and M. Selva, “Improving component substitution Pansharpening through multivariate regression of MS+Pan data,” 
        IEEE Transactions on Geoscience and Remote Sensing, vol. 45, no. 10, pp. 3230–3239, October 2007.
    [3] G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
        IEEE Transaction on Geoscience and Remote Sensing, 2014.
"""

import numpy as np
from utils import upsample_interp23

def GS(pan, hs):

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
    
    #sintetic intensity
    I = np.mean(u_hs, axis=2, keepdims=True)
    I0 = I-np.mean(I)
    
    image_hr = (pan-np.mean(pan))*(np.std(I0, ddof=1)/np.std(pan, ddof=1))+np.mean(I0)
    
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
    
    V_hat = V+ g*deltam
    
    I_GS = V_hat[:, :, 1:]
    
    I_GS = I_GS - np.mean(I_GS, axis=(0, 1))+means
    
    #adjustment
    I_GS[I_GS<0]=0
    I_GS[I_GS>1]=1
    
    return np.uint8(I_GS*255)
