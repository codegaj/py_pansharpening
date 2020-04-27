# -*- coding: utf-8 -*-
"""
License: MIT
@author: gaj
E-mail: anjing_guo@hnu.edu.cn
Paper References:
    [1] W. Carper, T. Lillesand, and R. Kiefer, “The use of Intensity-Hue-Saturation transformations for merging SPOT panchromatic and multispectral image data,” 
        Photogrammetric Engineering and Remote Sensing, vol. 56, no. 4, pp. 459–467, April 1990.
    [2] P. S. Chavez Jr., S. C. Sides, and J. A. Anderson, “Comparison of three different methods to merge multiresolution and multispectral data: Landsat TM and SPOT panchromatic,” 
        Photogrammetric Engineering and Remote Sensing, vol. 57, no. 3, pp. 295–303, March 1991.
    [3] T.-M. Tu, S.-C. Su, H.-C. Shyu, and P. S. Huang, “A new look at IHS-like image fusion methods,” 
        Information Fusion, vol. 2, no. 3, pp. 177–186, September 2001.
    [4] G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
        IEEE Transaction on Geoscience and Remote Sensing, 2014. 
"""

import numpy as np
from utils import upsample_interp23

def IHS(pan, hs):

    M, N, c = pan.shape
    m, n, C = hs.shape
    
    ratio = int(np.round(M/m))
        
    print('get sharpening ratio: ', ratio)
    assert int(np.round(M/m)) == int(np.round(N/n))
    
    #upsample
    u_hs = upsample_interp23(hs, ratio)
    
    I = np.mean(u_hs, axis=-1, keepdims=True)
    
    P = (pan - np.mean(pan))*np.std(I, ddof=1)/np.std(pan, ddof=1)+np.mean(I)
    
    I_IHS = u_hs + np.tile(P-I, (1, 1, C))
    
    #adjustment
    I_IHS[I_IHS<0]=0
    I_IHS[I_IHS>1]=1
    
    return np.uint8(I_IHS*255)
    