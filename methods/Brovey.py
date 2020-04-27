# -*- coding: utf-8 -*-
"""
License: MIT
@author: gaj
E-mail: anjing_guo@hnu.edu.cn
Paper References:
    [1] A. R. Gillespie, A. B. Kahle, and R. E. Walker, “Color enhancement of highly correlated images-II. Channel ratio and “Chromaticity” Transform techniques,” 
        Remote Sensing of Environment, vol. 22, no. 3, pp. 343–365, August 1987.
    [2] T.-M. Tu, S.-C. Su, H.-C. Shyu, and P. S. Huang, “A new look at IHS-like image fusion methods,” 
    Information Fusion, vol. 2, no. 3, pp. 177–186, September 2001.
    [3] G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
        IEEE Transaction on Geoscience and Remote Sensing, 2014. 
"""

import numpy as np
from utils import upsample_interp23

def Brovey(pan, hs):

    M, N, c = pan.shape
    m, n, C = hs.shape
    
    ratio = int(np.round(M/m))
        
    print('get sharpening ratio: ', ratio)
    assert int(np.round(M/m)) == int(np.round(N/n))
    
    #upsample
    u_hs = upsample_interp23(hs, ratio)
    
    I = np.mean(u_hs, axis=-1)
    
    image_hr = (pan-np.mean(pan))*(np.std(I, ddof=1)/np.std(pan, ddof=1))+np.mean(I)
    image_hr = np.squeeze(image_hr)

    I_Brovey=[]
    for i in range(C):
        temp = image_hr*u_hs[:, :, i]/(I+1e-8)
        temp = np.expand_dims(temp, axis=-1)
        I_Brovey.append(temp)
        
    I_Brovey = np.concatenate(I_Brovey, axis=-1) 
    
    #adjustment
    I_Brovey[I_Brovey<0]=0
    I_Brovey[I_Brovey>1]=1
    
    return np.uint8(I_Brovey*255)