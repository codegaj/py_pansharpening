# -*- coding: utf-8 -*-
"""
License: MIT
@author: gaj
E-mail: anjing_guo@hnu.edu.cn
Paper References:
    [1] King R L, Wang J. A wavelet based algorithm for pan sharpening Landsat 7 imagery
    [C]//IGARSS 2001. Scanning the Present and Resolving the Future. Proceedings. 
    IEEE 2001 International Geoscience and Remote Sensing Symposium (Cat. No. 01CH37217). IEEE, 2001, 2: 849-851.
"""

import numpy as np
from utils import upsample_interp23
import pywt

def Wavelet(pan, hs):

    M, N, c = pan.shape
    m, n, C = hs.shape
    
    ratio = int(np.round(M/m))
        
    print('get sharpening ratio: ', ratio)
    assert int(np.round(M/m)) == int(np.round(N/n))
    
    #upsample
    u_hs = upsample_interp23(hs, ratio)
    
    pan = np.squeeze(pan)
    pc = pywt.wavedec2(pan, 'haar', level=2)
    
    rec=[]
    for i in range(C):
        temp_dec = pywt.wavedec2(u_hs[:, :, i], 'haar', level=2)
        
        pc[0] = temp_dec[0]
        
        temp_rec = pywt.waverec2(pc, 'haar')
        temp_rec = np.expand_dims(temp_rec, -1)
        rec.append(temp_rec)
        
    I_Wavelet = np.concatenate(rec, axis=-1)
    
    #adjustment
    I_Wavelet[I_Wavelet<0]=0
    I_Wavelet[I_Wavelet>1]=1
    
    return np.uint8(I_Wavelet*255)
    