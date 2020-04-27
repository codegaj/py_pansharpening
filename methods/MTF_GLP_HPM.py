# -*- coding: utf-8 -*-
"""
License: MIT
@author: gaj
E-mail: anjing_guo@hnu.edu.cn
Paper References:
    [1] B. Aiazzi, L. Alparone, S. Baronti, and A. Garzelli, “Context-driven fusion of high spatial and spectral resolution images based on oversampled multiresolution analysis,” 
        IEEE Transactions on Geoscience and Remote Sensing, vol. 40, no. 10, pp. 2300–2312, October 2002.
    [2] B. Aiazzi, L. Alparone, S. Baronti, A. Garzelli, and M. Selva, “MTF-tailored multiscale fusion of high-resolution MS and Pan imagery,”
        Photogrammetric Engineering and Remote Sensing, vol. 72, no. 5, pp. 591–596, May 2006.
    [3] G. Vivone, R. Restaino, M. Dalla Mura, G. Licciardi, and J. Chanussot, “Contrast and error-based fusion schemes for multispectral image pansharpening,” 
        IEEE Geoscience and Remote Sensing Letters, vol. 11, no. 5, pp. 930–934, May 2014.
    [4] G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
        IEEE Transaction on Geoscience and Remote Sensing, 2014.
"""

import numpy as np
from utils import upsample_interp23
import cv2
from scipy import signal

def gaussian2d(N, std):
    
    t=np.arange(-(N-1)/2,(N+2)/2)
    t1,t2=np.meshgrid(t,t)
    std=np.double(std)
    w = np.exp(-0.5*(t1/std)**2)*np.exp(-0.5*(t2/std)**2) 
    return w
    
def kaiser2d(N, beta):
    
    t=np.arange(-(N-1)/2,(N+1)/2)/np.double(N-1)
    t1,t2=np.meshgrid(t,t)
    t12=np.sqrt(t1*t1+t2*t2)
    w1=np.kaiser(N,beta)
    w=np.interp(t12,t,w1)
    w[t12>t[-1]]=0
    w[t12<t[0]]=0
    
    return w

def fir_filter_wind(Hd,w):
    """
	compute fir filter with window method
	Hd: 	desired freqeuncy response (2D)
	w: 		window (2D)
	"""
	
    hd=np.rot90(np.fft.fftshift(np.rot90(Hd,2)),2)
    h=np.fft.fftshift(np.fft.ifft2(hd))
    h=np.rot90(h,2)
    h=h*w
    h=h/np.sum(h)
    
    return h

def MTF_GLP_HPM(pan, hs, sensor='gaussian'):
    
    M, N, c = pan.shape
    m, n, C = hs.shape
    
    ratio = int(np.round(M/m))
        
    print('get sharpening ratio: ', ratio)
    assert int(np.round(M/m)) == int(np.round(N/n))
    
    #upsample
    u_hs = upsample_interp23(hs, ratio)
    
    #equalization
    image_hr = np.tile(pan, (1, 1, C))
    
    image_hr = (image_hr - np.mean(image_hr, axis=(0,1)))*(np.std(u_hs, axis=(0, 1), ddof=1)/np.std(image_hr, axis=(0, 1), ddof=1))+np.mean(u_hs, axis=(0,1))
    
    pan_lp = np.zeros_like(u_hs)
    N =31
    fcut = 1/ratio
    match = 0
    
    if sensor == 'gaussian':
        sig = (1/(2*(2.772587)/ratio**2))**0.5
        kernel = np.multiply(cv2.getGaussianKernel(9, sig), cv2.getGaussianKernel(9,sig).T)
        
        t=[]
        for i in range(C):
            temp = signal.convolve2d(image_hr[:, :, i], kernel, mode='same', boundary = 'wrap')
            temp = temp[0::ratio, 0::ratio]
            temp = np.expand_dims(temp, -1)
            t.append(temp)
        
        t = np.concatenate(t, axis=-1)
        pan_lp = upsample_interp23(t, ratio)
    
    elif sensor == None:
        match=1
        GNyq = 0.3*np.ones((C,))
    elif sensor=='QB':
        match=1
        GNyq = np.asarray([0.34, 0.32, 0.30, 0.22],dtype='float32')    # Band Order: B,G,R,NIR
    elif sensor=='IKONOS':
        match=1           #MTF usage
        GNyq = np.asarray([0.26,0.28,0.29,0.28],dtype='float32')    # Band Order: B,G,R,NIR
    elif sensor=='GeoEye1':
        match=1             # MTF usage
        GNyq = np.asarray([0.23,0.23,0.23,0.23],dtype='float32')    # Band Order: B,G,R,NIR   
    elif sensor=='WV2':
        match=1            # MTF usage
        GNyq = [0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.27]
    elif sensor=='WV3':
        match=1             #MTF usage
        GNyq = 0.29 * np.ones(8)
    
    if match==1:
        t = []
        for i in range(C):
            alpha = np.sqrt(N*(fcut/2)**2/(-2*np.log(GNyq)))
            H = np.multiply(cv2.getGaussianKernel(N, alpha[i]), cv2.getGaussianKernel(N, alpha[i]).T)
            HD = H/np.max(H)
            
            h = fir_filter_wind(HD, kaiser2d(N, 0.5))
            
            temp = signal.convolve2d(image_hr[:, :, i], np.real(h), mode='same', boundary = 'wrap')
            temp = temp[0::ratio, 0::ratio]
            temp = np.expand_dims(temp, -1)
            t.append(temp)
        
        t = np.concatenate(t, axis=-1)
        pan_lp = upsample_interp23(t, ratio)
        
    I_MTF_GLP_HPM = u_hs*(image_hr/(pan_lp+1e-8))      
    
    #adjustment
    I_MTF_GLP_HPM[I_MTF_GLP_HPM<0]=0
    I_MTF_GLP_HPM[I_MTF_GLP_HPM>1]=1
    
    return np.uint8(I_MTF_GLP_HPM*255)