# -*- coding: utf-8 -*-
"""
License: MIT
@author: gaj
E-mail: anjing_guo@hnu.edu.cn
"""

import cv2
import numpy as np
from scipy import ndimage
from scipy import signal
import scipy.misc as misc

def upsample_bilinear(image, ratio):
    
    h,w,c = image.shape
    re_image = cv2.resize(image, (w*ratio, h*ratio), cv2.INTER_LINEAR)
    
    return re_image

def upsample_bicubic(image, ratio):
    
    h,w,c = image.shape
    re_image = cv2.resize(image, (w*ratio, h*ratio), cv2.INTER_CUBIC)
    
    return re_image

def upsample_interp23(image, ratio):

    image = np.transpose(image, (2, 0, 1))
    
    b,r,c = image.shape

    CDF23 = 2*np.array([0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0, -0.005192756653, 0, 0.000807762146, 0, -0.000060081482])
    d = CDF23[::-1] 
    CDF23 = np.insert(CDF23, 0, d[:-1])
    BaseCoeff = CDF23
    
    first = 1
    for z in range(1,np.int(np.log2(ratio))+1):
        I1LRU = np.zeros((b, 2**z*r, 2**z*c))
        if first:
            I1LRU[:, 1:I1LRU.shape[1]:2, 1:I1LRU.shape[2]:2]=image
            first = 0
        else:
            I1LRU[:,0:I1LRU.shape[1]:2,0:I1LRU.shape[2]:2]=image
        
        for ii in range(0,b):
            t = I1LRU[ii,:,:]
            for j in range(0,t.shape[0]):
                t[j,:]=ndimage.correlate(t[j,:],BaseCoeff,mode='wrap')
            for k in range(0,t.shape[1]):
                t[:,k]=ndimage.correlate(t[:,k],BaseCoeff,mode='wrap')
            I1LRU[ii,:,:]=t
        image = I1LRU
        
    re_image=np.transpose(I1LRU, (1, 2, 0))
        
    return re_image

def upsample_mat_interp23(image, ratio=4):
    
    '''2 pixel shift compare with original matlab version'''
    
    shift=2
        
    h,w,c = image.shape
    
    basecoeff = np.array([[-4.63495665e-03, -3.63442646e-03,  3.84904063e-18,
     5.76678319e-03,  1.08358664e-02,  1.01980790e-02,
    -9.31747402e-18, -1.75033181e-02, -3.17660068e-02,
    -2.84531643e-02,  1.85181518e-17,  4.42450253e-02,
     7.71733386e-02,  6.70554910e-02, -2.85299239e-17,
    -1.01548683e-01, -1.78708388e-01, -1.60004642e-01,
     3.61741232e-17,  2.87940558e-01,  6.25431459e-01,
     8.97067600e-01,  1.00107877e+00,  8.97067600e-01,
     6.25431459e-01,  2.87940558e-01,  3.61741232e-17,
    -1.60004642e-01, -1.78708388e-01, -1.01548683e-01,
    -2.85299239e-17,  6.70554910e-02,  7.71733386e-02,
     4.42450253e-02,  1.85181518e-17, -2.84531643e-02,
    -3.17660068e-02, -1.75033181e-02, -9.31747402e-18,
     1.01980790e-02,  1.08358664e-02,  5.76678319e-03,
     3.84904063e-18, -3.63442646e-03, -4.63495665e-03]])
    
    coeff = np.dot(basecoeff.T, basecoeff)
    
    I1LRU = np.zeros((ratio*h, ratio*w, c))
    
    I1LRU[shift::ratio, shift::ratio, :]=image
    
    for i in range(c):
        temp = I1LRU[:, :, i]
        temp = ndimage.convolve(temp, coeff, mode='wrap')
        I1LRU[:, :, i]=temp
        
    return I1LRU

def gaussian2d (N, std):
    
    t=np.arange(-(N-1)/2,(N+2)/2)
    t1,t2=np.meshgrid(t,t)
    std=np.double(std)
    w = np.exp(-0.5*(t1/std)**2)*np.exp(-0.5*(t2/std)**2) 
    return w
    
def kaiser2d (N, beta):
    
    t=np.arange(-(N-1)/2,(N+2)/2)/np.double(N-1)
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

def downgrade_images(I_MS, I_PAN, ratio, sensor=None):
    """
    downgrade MS and PAN by a ratio factor with given sensor's gains
    """
    I_MS=np.double(I_MS)
    I_PAN=np.double(I_PAN)
    
    I_MS = np.transpose(I_MS, (2, 0, 1))
    I_PAN = np.squeeze(I_PAN)
    
    ratio=np.double(ratio)
    flag_PAN_MTF=0
    
    if sensor=='QB':
        flag_resize_new = 2
        GNyq = np.asarray([0.34, 0.32, 0.30, 0.22],dtype='float32')    # Band Order: B,G,R,NIR
        GNyqPan = 0.15
    elif sensor=='IKONOS':
        flag_resize_new = 2             #MTF usage
        GNyq = np.asarray([0.26,0.28,0.29,0.28],dtype='float32')    # Band Order: B,G,R,NIR
        GNyqPan = 0.17;
    elif sensor=='GeoEye1':
        flag_resize_new = 2             # MTF usage
        GNyq = np.asarray([0.23,0.23,0.23,0.23],dtype='float32')    # Band Order: B,G,R,NIR
        GNyqPan = 0.16     
    elif sensor=='WV2':
        flag_resize_new = 2             # MTF usage
        GNyq = [0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.27]
        GNyqPan = 0.11
    elif sensor=='WV3':
        flag_resize_new = 2             #MTF usage
        GNyq = 0.29 * np.ones(8)
        GNyqPan = 0.15
    else:
        '''the default way'''
        flag_resize_new = 1
    
    '''the default downgrading method is gaussian'''
    if flag_resize_new == 1:
        
#        I_MS_LP = np.zeros((I_MS.shape[0],int(np.round(I_MS.shape[1]/ratio)+ratio),int(np.round(I_MS.shape[2]/ratio)+ratio)))
#            
#        for idim in range(I_MS.shape[0]):
#            imslp_pad=np.pad(I_MS[idim,:,:],int(2*ratio),'symmetric')
#            I_MS_LP[idim,:,:]=misc.imresize(imslp_pad,1/ratio,'bicubic',mode='F')
#            
#        I_MS_LR = I_MS_LP[:,2:-2,2:-2]
#       
#        I_PAN_pad=np.pad(I_PAN,int(2*ratio),'symmetric')
#        I_PAN_LR=misc.imresize(I_PAN_pad,1/ratio,'bicubic',mode='F')
#        I_PAN_LR=I_PAN_LR[2:-2,2:-2]
        
        sig = (1/(2*(2.772587)/ratio**2))**0.5
        kernel = np.multiply(cv2.getGaussianKernel(9, sig), cv2.getGaussianKernel(9,sig).T)
        
        t=[]
        for i in range(I_MS.shape[0]):
            temp = signal.convolve2d(I_MS[i, :, :], kernel, mode='same', boundary = 'wrap')
            temp = temp[0::int(ratio), 0::int(ratio)]
            temp = np.expand_dims(temp, 0)
            t.append(temp)
            
        I_MS_LR = np.concatenate(t, axis=0)
        
        I_PAN_LR = signal.convolve2d(I_PAN, kernel, mode='same', boundary = 'wrap')
        I_PAN_LR = I_PAN_LR[0::int(ratio), 0::int(ratio)]
        
    elif flag_resize_new==2:
        
        N=41
        I_MS_LP=np.zeros(I_MS.shape)
        fcut=1/ratio
        
        for j in range(I_MS.shape[0]):
            #fir filter with window method
            alpha = np.sqrt(((N-1)*(fcut/2))**2/(-2*np.log(GNyq[j])))
            H=gaussian2d(N,alpha)
            Hd=H/np.max(H)
            w=kaiser2d(N,0.5)
            h=fir_filter_wind(Hd,w)
            I_MS_LP[j,:,:] = ndimage.filters.correlate(I_MS[j,:,:],np.real(h),mode='nearest')
        
        if flag_PAN_MTF==1:
            #fir filter with window method
            alpha = np.sqrt(((N-1)*(fcut/2))**2/(-2*np.log(GNyqPan)))
            H=gaussian2d(N,alpha)
            Hd=H/np.max(H)
            h=fir_filter_wind(Hd,w)
            I_PAN = ndimage.filters.correlate(I_PAN,np.real(h),mode='nearest')
            I_PAN_LR=I_PAN[int(ratio/2):-1:int(ratio),int(ratio/2):-1:int(ratio)]
            
        else:
            #bicubic resize
            I_PAN_pad=np.pad(I_PAN,int(2*ratio),'symmetric')
            I_PAN_LR=misc.imresize(I_PAN_pad,1/ratio,'bicubic',mode='F')
            I_PAN_LR=I_PAN_LR[2:-2,2:-2]
            
        I_MS_LR=I_MS_LP[:,int(ratio/2):-1:int(ratio),int(ratio/2):-1:int(ratio)]     
        
    I_MS_LR = np.transpose(I_MS_LR, (1, 2, 0))
    I_PAN_LR = np.expand_dims(I_PAN_LR, -1)
    
    return I_MS_LR,I_PAN_LR