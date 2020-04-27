# -*- coding: utf-8 -*-
"""
License: MIT
@author: gaj
E-mail: anjing_guo@hnu.edu.cn
Paper References:
    [1] P. S. Chavez Jr. and A. W. Kwarteng, “Extracting spectral contrast in Landsat Thematic Mapper image data using selective principal component analysis,” 
        Photogrammetric Engineering and Remote Sensing, vol. 55, no. 3, pp. 339–348, March 1989.
    [2] G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
        IEEE Transaction on Geoscience and Remote Sensing, 2014.
"""

import numpy as np
from utils import upsample_interp23
from sklearn.decomposition import PCA as princomp

def PCA(pan, hs):

    M, N, c = pan.shape
    m, n, C = hs.shape
    
    ratio = int(np.round(M/m))
        
    print('get sharpening ratio: ', ratio)
    assert int(np.round(M/m)) == int(np.round(N/n))
    
    image_hr = pan
    
    #upsample
    u_hs = upsample_interp23(hs, ratio)
    
    p = princomp(n_components=C)
    pca_hs = p.fit_transform(np.reshape(u_hs, (M*N, C)))
    
    pca_hs = np.reshape(pca_hs, (M, N, C))
    
    I = pca_hs[:, :, 0]
    
    image_hr = (image_hr - np.mean(image_hr))*np.std(I, ddof=1)/np.std(image_hr, ddof=1)+np.mean(I)
    
    pca_hs[:, :, 0] = image_hr[:, :, 0]
    
    I_PCA = p.inverse_transform(pca_hs)
    
    #equalization
    I_PCA = I_PCA-np.mean(I_PCA, axis=(0, 1))+np.mean(u_hs)
    
    #adjustment
    I_PCA[I_PCA<0]=0
    I_PCA[I_PCA>1]=1
    
    return np.uint8(I_PCA*255)