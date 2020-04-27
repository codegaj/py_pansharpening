# -*- coding: utf-8 -*-
"""
License: MIT
@author: gaj
E-mail: anjing_guo@hnu.edu.cn 
"""

import numpy as np
from utils import upsample_bicubic

def Bicubic(pan, hs):

    M, N, c = pan.shape
    m, n, C = hs.shape
    
    ratio = int(np.round(M/m))
        
    print('get sharpening ratio: ', ratio)
    assert int(np.round(M/m)) == int(np.round(N/n))
    
    # jsut upsample with bicubic
    I_Bicubic = upsample_bicubic(hs, ratio)
    
    #adjustment
    I_Bicubic[I_Bicubic<0]=0
    I_Bicubic[I_Bicubic>1]=1
    
    return np.uint8(I_Bicubic*255)