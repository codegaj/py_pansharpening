# -*- coding: utf-8 -*-
"""
License: GNU-3.0
Referenc: http://www.naotoyokoya.com/
Paper References:
    [1] N. Yokoya, T. Yairi, and A. Iwasaki, "Coupled nonnegative matrix factorization unmixing for hyperspectral and multispectral data fusion," 
        IEEE Trans. Geosci. Remote Sens., vol. 50, no. 2, pp. 528-537, 2012.
    [2] N. Yokoya, T. Yairi, and A. Iwasaki, "Hyperspectral, multispectral, and panchromatic data fusion based on non-negative matrix factorization," 
        Proc. WHISPERS, Lisbon, Portugal, Jun. 6-9, 2011.
    [3] N. Yokoya, N. Mayumi, and A. Iwasaki, "Cross-calibration for data fusion of EO-1/Hyperion and Terra/ASTER," 
        IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens., vol. 6, no. 2, pp. 419-426, 2013.
"""

import numpy as np
from scipy.special import erfinv

def CNMF(MSI, HSI, mask=0, verbose='off',MEMs=0):
    '''
    COUPLED NONNEGATIVE MATRIX FACTORIZATION (CNMF)

    Copyright (c) 2016 Naoto Yokoya
    Email: yokoya@sal.rcast.u-tokyo.ac.jp
    Update: 2016/04/01

    References:
    [1] N. Yokoya, T. Yairi, and A. Iwasaki, "Coupled nonnegative matrix
        factorization unmixing for hyperspectral and multispectral data fusion,"
        IEEE Trans. Geosci. Remote Sens., vol. 50, no. 2, pp. 528-537, 2012.
    [2] N. Yokoya, N. Mayumi, and A. Iwasaki, "Cross-calibration for data fusion
        of EO-1/Hyperion and Terra/ASTER," IEEE J. Sel. Topics Appl. Earth Observ.
        Remote Sens., vol. 6, no. 2, pp. 419-426, 2013.
    [3] N. Yokoya, T. Yairi, and A. Iwasaki, "Hyperspectral, multispectral,
        and panchromatic data fusion based on non-negative matrix factorization,"
        Proc. WHISPERS, Lisbon, Portugal, Jun. 6-9, 2011.

    USAGE
        Out = CNMF_fusion(HSI,MSI,mask,verbose)

    INPUT
        HSI     : Low-spatial-resolution HS image (rows2,cols2,bands2)
        MSI     : MS image (rows1,cols1,bands1)
        mask    : (optional) Binary mask for processing (rows2,cols2) (0: mask, 1: image)
        verbose : (optional) Print out processing status
        MEMs    : (optional) Manually defined endmembers (bands2, num. of endmembers)

    OUTPUT
        Out : High-spatial-resolution HS image (rows1,cols1,bands2)
    '''

    # masking mode
    if np.isscalar(mask):
        masking = 0
    else:
        masking = 1

    # image size
    rows1 = MSI.shape[0]
    cols1 = MSI.shape[1]
    bands1 = MSI.shape[2]
    rows2 = HSI.shape[0]
    cols2 = HSI.shape[1]
    bands2 = HSI.shape[2]

    w = int(rows1/rows2)

    # Estimation of R
    if verbose == 'on':
        print('Estimate R...')
    R = estR(HSI,MSI,mask)
    for b in range(bands1):
        msi = MSI[:,:,b].reshape(rows1,cols1).copy()
        msi = msi - R[b,-1]
        msi[np.nonzero(msi<0)] = 0
        MSI[:,:,b] = msi.copy()
    R = R[:,0:bands2]

    # parameters
    th_h = 1e-8 # Threshold of change ratio in inner loop for HS unmixing
    th_m = 1e-8 # Threshold of change ratio in inner loop for MS unmixing
    th2 = 1e-2 # Threshold of change ratio in outer loop
    sum2one = 2*( MSI.mean()/0.7455)**0.5 / bands1**3 # Parameter of sum to 1 constraint

    if bands1 == 1:
        I1 = 75 # Maximum iteration of inner loop
        I2 = 1 # Maximum iteration of outer loop
    else:
        I1 = 200 # Maximum iteration of inner loop (200-300)
        I2 = 1 # Maximum iteration of outer loop (1-3)

    # initialization of H_hyper
    # 0: constant (fast)
    # 1: nonnegative least squares (slow)
    init_mode = 0

    # avoid nonnegative values
    HSI[np.nonzero(HSI<0)] = 0
    MSI[np.nonzero(MSI<0)] = 0

    if masking == 0:
        HSI = HSI.reshape(rows2*cols2,bands2).transpose()
        MSI = MSI.reshape(rows1*cols1,bands1).transpose()
    else:
        HSI = HSI.reshape(rows2*cols2,bands2)
        MSI = MSI.reshape(rows1*cols1,bands1)

        mask2 = zoom_nn(mask,w)
        HSI = HSI[mask.reshape(rows2*cols2)==1,:].transpose()
        MSI = MSI[mask2.reshape(rows1*cols1)==1,:].transpose()

    # manually define endmembers
    if np.isscalar(MEMs) == False:
        if MEMs.shape[0] == bands2 and len(MEMs.shape) == 2:
            M_m = MEMs.shape[1]
        else:
            print('Please check the size of manually defined endmembers.')
            M_m = 0
            MEMs = 0
    else:
        M_m = 0

    # number of endmembers
    M_est = int(round(vd(HSI,5*10**-2)))
    M = max([min([30,bands2]), M_est]) # M can be automatically defined, for example, by VD
    if verbose == 'on':
        print('Number of endmembers: ', M+M_m)

    # CNMF Initializatioin
    HSI, MSI, W_hyper, H_hyper, W_multi, H_multi, RMSE_h, RMSE_m = CNMF_init(rows1,cols1,w,M,HSI,MSI,sum2one,I1,th_h,th_m,R,init_mode,mask,verbose,MEMs)

    cost = np.zeros((2,I2+1))
    cost[0,0] = RMSE_h
    cost[1,0] = RMSE_m

    # CNMF Iteration
    for i in range(I2):
        W_hyper, H_hyper, W_multi1, H_multi1, W_multi2, H_multi2, RMSE_h, RMSE_m = CNMF_ite(rows1,cols1,w,M+M_m,HSI,MSI,W_hyper,H_hyper,W_multi,H_multi,I1,th_h,th_m,I2,i,R,mask,verbose)

        cost[0,i+1] = RMSE_h
        cost[1,i+1] = RMSE_m

        if (cost[0,i]-cost[0,i+1])/cost[0,i]>th2 and (cost[1,i]-cost[1,i+1])/cost[1,i]>th2 and i<I2-1:
            W_multi = W_multi2.copy()
            H_multi = H_multi2.copy()
        elif i == I2-1:
            if verbose == 'on':
                print('Max outer interation.')
        else:
            if verbose == 'on':
                print('END')
            break

    if masking == 0:
        Out = np.dot(W_hyper[0:bands2,:] , H_multi ).transpose().reshape(rows1,cols1,bands2)
    else:
        Out = np.zeros((rows1*cols1,bands2))
        Out[mask2.reshape(rows1*cols1)==1,:] = np.dot(W_hyper[0:bands2,:] , H_multi ).transpose()
        Out = Out.reshape(rows1,cols1,bands2)

    #adjustment, 2020/4/13
    Out[Out<0]=0
    Out[Out>1]=1
    
    return np.uint8(Out*255)


def CNMF_init(xdata,ydata,w,M,hyper,multi,delta,I_in,delta_h,delta_m,srf,init_mode=0,mask=0,verbose='off',MEMs=0):
    '''
    COUPLED NONNEGATIVE MATRIX FACTORIZATION (CNMF)

    Copyright (c) 2016 Naoto Yokoya
    Email: yokoya@sal.rcast.u-tokyo.ac.jp
    Update: 2016/04/01

    References:
    [1] N. Yokoya, T. Yairi, and A. Iwasaki, "Coupled nonnegative matrix
        factorization unmixing for hyperspectral and multispectral data fusion,"
        IEEE Trans. Geosci. Remote Sens., vol. 50, no. 2, pp. 528-537, 2012.
    [2] N. Yokoya, T. Yairi, and A. Iwasaki, "Hyperspectral, multispectral,
        and panchromatic data fusion based on non-negative matrix factorization,"
        Proc. WHISPERS, Lisbon, Portugal, Jun. 6-9, 2011.

    This function is the initilization function of CNMF.

    USAGE
        hyper, multi, W_hyper, H_hyper, W_multi, H_multi, RMSE_h, RMSE_m =
        CNMF_init(xdata,ydata,w,M,hyper,multi,delta,I_in,delta_h,delta_m,srf,init_mode,mask,verbose)

    INPUT
        xdata           : image height
        ydata           : image width
        w               : multiple difference of ground sampling distance (scalar)
        M               : Number of endmembers
        hyper           : Low-spatial-resolution HS image (band, xdata/w*ydata/w)
        multi           : MS image (multi_band, xdata*ydata)
        delta           : Parameter of sum to one constraint
        I_in            : Maximum number of inner iteration
        delta_h         : Parameter for HS unmixing
        delta_m         : Parameter for MS unmixing
        srf             : Relative specctral response function
        init_mode       : Initialization mode (0: const, 1: nnls)
        mask            : (optional) Binary mask for processing (xdata/w,ydata/w)
        verbose         : (optional) Print out processing status
        MEMs            : (optional) Manually defined endmembers (bands2, num. of endmembers)

    OUTPUT
        hyper       : Low-spatial-resolution HS image with ones (band+1, xdata/w*ydata/w)
        multi       : MS image with ones (multi_band+1, xdata*ydata)
        W_hyper     : HS endmember matrix with ones (band+1, M)
        H_hyper     : HS abundance matrix (M, xdata/w*ydata/w)
        W_multi     : MS endmember matrix with ones (multi_band+1, M)
        H_multi     : MS abundance matrix (M, xdata*ydata)
        RMSE_h      : RMSE of HS unmixing
        RMSE_m      : RMSE of MS unmixing
    '''

    MIN_MS_BANDS = 3

    band = np.size(hyper,0)
    multi_band = np.size(multi,0)
    hx = int(xdata/w)
    hy = int(ydata/w)
    if verbose == 'on':
        print('Initialize Wh by VCA')
    W_hyper, indices = vca( hyper, M )

    # Add manually defined endmembers
    if np.isscalar(MEMs) == False:
        W_hyper = np.hstack((W_hyper, MEMs))
        M = W_hyper.shape[1]

    # masking mode
    if np.isscalar(mask):
        masking = 0
        mask = np.ones((hy,hx))
    else:
        masking = 1

    # Initialize H_hyper: (M, N_h)
    if masking == 0:
        H_hyper = np.ones((M, hx*hy))/M
    else:
        H_hyper = np.ones((M, hx*hy))/M
        H_hyper = H_hyper[:,mask.reshape(hx*hy)==1]

    if init_mode == 1:
        if verbose == 'on':
            print('Initialize Hh by NLS')
        # initialize H_hyper by nonnegative least squares
        H_hyper = nls_su(hyper,W_hyper)

    # Sum-to-one constraint
    W_hyper = np.vstack((W_hyper, delta*np.ones((1,np.size(W_hyper, 1)))))
    hyper = np.vstack((hyper, delta*np.ones((1,np.size(hyper, 1)))))

    # NMF for Vh 1st
    if verbose == 'on':
        print ('NMF for Vh ( 1 )')
    for i in range(I_in):
        # Initialization of H_hyper
        if i == 0:
            cost0 = 0
            for q in range(I_in*3):
                # Update H_hyper
                H_hyper_old = H_hyper
                H_hyper_n = np.dot(W_hyper.transpose(), hyper)
                H_hyper_d = np.dot(np.dot(W_hyper.transpose(), W_hyper), H_hyper)
                H_hyper = (H_hyper*H_hyper_n)/H_hyper_d
                cost = np.sum((hyper[0:band, :] - np.dot(W_hyper[0:band, :], H_hyper))**2)
                if q > 1 and (cost0-cost)/cost < delta_h:
                    if verbose == 'on':
                        print('Initialization of H_hyper converged at the ', q, 'th iteration ')
                    H_hyper = H_hyper_old
                    break
                cost0 = cost
        else:
            # Update W_hyper
            W_hyper_old = W_hyper
            W_hyper_n = np.dot(hyper[0:band, :], (H_hyper.transpose()))
            W_hyper_d = np.dot(np.dot(W_hyper[0:band,:], H_hyper), H_hyper.transpose())
            W_hyper[0:band, :] = (W_hyper[0:band, :]*W_hyper_n)/W_hyper_d
            # Update H_hyper
            H_hyper_old = H_hyper
            H_hyper_n = np.dot(W_hyper.transpose(), hyper)
            H_hyper_d = np.dot(np.dot(W_hyper.transpose(), W_hyper), H_hyper)
            H_hyper = (H_hyper*H_hyper_n)/H_hyper_d
            cost = np.sum((hyper[0:band, :] - np.dot(W_hyper[0:band, :], H_hyper))**2)
            if (cost0-cost)/cost < delta_h:
                if verbose == 'on':
                    print('Optimization of HS unmixing converged at the ', i, 'th iteration ')
                W_hyper = W_hyper_old
                H_hyper = H_hyper_old
                break
            cost0 = cost

    RMSE_h = (cost0/(hyper.shape[1]*band))**0.5
    if verbose == 'on':
        print('    RMSE(Vh) = ', RMSE_h)

    # initialize W_multi: (multi_band, M)
    W_multi = np.dot(srf, W_hyper[0:band,:])
    W_multi = np.vstack((W_multi, delta*np.ones((1, M))))
    multi = np.vstack((multi, delta*np.ones((1, multi.shape[1]))))

    # initialize H_multi by interpolation
    if masking == 0:
        H_multi = np.ones((M, xdata*ydata))/M
        for i in range(M):
            tmp = zoom_bi(H_hyper[i,:].reshape(hx,hy).copy(),w)
            H_multi[i,:] = tmp.reshape(1,xdata*ydata)
        H_multi[np.nonzero(H_multi<0)] = 0
    else:
        mask2 = zoom_nn(mask,w)
        H_multi = np.ones((M,multi.shape[1]))/M
        for i in range(M):
            tmp = np.zeros((hx,hy))
            tmp[np.nonzero(mask>0)] = H_hyper[i,:].copy()
            tmp = zoom_bi(tmp,w)
            H_multi[i,:] = tmp[np.nonzero(mask2>0)].copy()
        H_multi[np.nonzero(H_multi<0)] = 0

    # NMF for Vm 1st
    if verbose == 'on':
        print('NMF for Vm ( 1 )')
    for i in range(I_in):
        if i == 0:
            cost0 = 0
            for q in range(I_in):
                # Update H_multi
                H_multi_old = H_multi
                H_multi_n = np.dot(W_multi.transpose(), multi)
                H_multi_d = np.dot(np.dot(W_multi.transpose(), W_multi), H_multi)
                H_multi = (H_multi*H_multi_n)/H_multi_d
                cost = np.sum((multi[0:multi_band, :] - np.dot(W_multi[0:multi_band, :], H_multi))**2)
                if q > 1 and (cost0-cost)/cost < delta_m:
                    if verbose == 'on':
                        print('Initialization of H_multi converged at the ', q, 'th iteration ')
                    H_multi = H_multi_old
                    break
                cost0 = cost
        else:
            # Update W_multi
            W_multi_old = W_multi
            if multi_band > MIN_MS_BANDS:
                W_multi_n = np.dot(multi[0:multi_band, :], H_multi.transpose())
                W_multi_d = np.dot(np.dot(W_multi[0:multi_band, :], H_multi), H_multi.transpose())
                W_multi[0:multi_band, :] = (W_multi[0:multi_band, :]*W_multi_n)/W_multi_d
            # Update H_hyper
            H_multi_old = H_multi
            H_multi_n = np.dot(W_multi.transpose(), multi)
            H_multi_d = np.dot(np.dot(W_multi.transpose(), W_multi), H_multi)
            H_multi = H_multi*H_multi_n/H_multi_d
            cost = np.sum((multi[0:multi_band, :]-np.dot(W_multi[0:multi_band, :], H_multi))**2)
            if (cost0-cost)/cost < delta_m:
                if verbose == 'on':
                    print('Optimization of MS unmixing converged at the ', i, 'th iteration ')
                W_multi = W_multi_old
                H_multi = H_multi_old
                break
            cost0=cost

    RMSE_m = (cost0/((multi.shape[1])*multi_band))**0.5
    if verbose == 'on':
        print('    RMSE(Vm) = ', RMSE_m) # MSE(Mean Squared Error) in NMF of Vm

    return hyper, multi, W_hyper, H_hyper, W_multi, H_multi, RMSE_h, RMSE_m

def CNMF_ite(xdata,ydata,w,M,hyper,multi,W_hyper,H_hyper,W_multi,H_multi,I_in,delta_h,delta_m,I_out,i_out,srf,mask=0,verbose='off'):
    '''
    COUPLED NONNEGATIVE MATRIX FACTORIZATION (CNMF)

    Copyright (c) 2016 Naoto Yokoya
    Email: yokoya@sal.rcast.u-tokyo.ac.jp
    Update: 2016/04/01

    References:
    [1] N. Yokoya, T. Yairi, and A. Iwasaki, "Coupled nonnegative matrix
        factorization unmixing for hyperspectral and multispectral data fusion,"
        IEEE Trans. Geosci. Remote Sens., vol. 50, no. 2, pp. 528-537, 2012.
    [2] N. Yokoya, T. Yairi, and A. Iwasaki, "Hyperspectral, multispectral,
        and panchromatic data fusion based on non-negative matrix factorization,"
        Proc. WHISPERS, Lisbon, Portugal, Jun. 6-9, 2011.

    This function is the iteration function of CNMF.

    USAGE
        W_hyper, H_hyper, W_multi1, H_multi1, W_multi2, H_multi2, RMSE_h, RMSE_m =
        CNMF_ite(xdata,ydata,w,M,hyper,multi,W_hyper,H_hyper,W_multi,H_multi,ite_max,delta_h,delta_m,iter,srf,mask,verbose)

    INPUT
        xdata           : image height
        ydata           : image width
        w               : multiple difference of ground sampling distance (scalar)
        M               : Number of endmembers
        hyper           : Low-spatial-resolution HS image (band, xdata/w*ydata/w)
        multi           : MS image (multi_band, xdata*ydata)
        W_hyper         : HS endmember matrix with ones (band+1, M)
        H_hyper         : HS abundance matrix (M, xdata/w*ydata/w)
        W_multi         : MS endmember matrix with ones (multi_band+1, M)
        H_multi         : MS abundance matrix (M, xdata*ydata)
        delta           : Parameter of sum to one constraint
        I_in            : Maximum number of inner iteration
        delta_h         : Parameter for HS unmixing
        delta_m         : Parameter for MS unmixing
        I_out           : Maximum number of outer iteration
        i_out           : Current number of outer iteration
        srf             : Relative specctral response function
        mask            : (optional) Binary mask for processing (xdata/w,ydata/w)

    OUTPUT
        W_hyper     : HS endmember matrix with ones (band+1, M)
        H_hyper     : HS abundance matrix (M, xdata/w*ydata/w)
        W_multi1    : MS endmember matrix with ones before MS unmixing (multi_band+1, M)
        H_multi1    : MS abundance matrix before MS unmixing (M, xdata*ydata)
        W_multi2    : MS endmember matrix with ones after MS unmixing (multi_band+1, M)
        H_multi2    : MS abundance matrix after MS unmixing (M, xdata*ydata)
        RMSE_h      : RMSE of HS unmixing
        RMSE_m      : RMSE of MS unmixing
    '''

    MIN_MS_BANDS = 3

    band = np.size(hyper,0)-1
    multi_band = np.size(multi,0)-1
    hx = int(xdata/w)
    hy = int(ydata/w)

    # masking mode
    if np.isscalar(mask):
        masking = 0
        mask = np.ones((hy,hx))
    else:
        masking = 1

    if verbose == 'on':
        print('Iteration', i_out)

    # Initialize H_hyper form H_multi
    if masking == 0:
        H_hyper = gaussian_down_sample(H_multi.transpose().reshape(xdata,ydata,M),w).reshape(hx*hy,M).transpose()
    else:
        mask2 = zoom_nn(mask,w)
        for q in range(M):
            tmp = np.zeros((xdata,ydata))
            tmp[mask2>0] = H_multi[q,:].copy()
            tmp = gaussian_down_sample(tmp.reshape(xdata,ydata,1),w).reshape(hx,hy)
            H_hyper[q,:] = tmp[mask>0].copy().reshape(1,mask.sum())

    # NMF for Vh
    if verbose == 'on':
        print('NMF for Vh (', i_out+2, ')')
    for i in range(I_in):
        if i == 0:
            cost0 = 0
            for q in range(I_in):
                # Update W_hyper
                W_hyper_old = W_hyper
                W_hyper_n = np.dot(hyper[0:band, :], H_hyper.transpose())
                W_hyper_d = np.dot(np.dot(W_hyper[0:band, :], H_hyper), H_hyper.transpose())
                W_hyper[0:band, :] = (W_hyper[0:band, :]*W_hyper_n)/W_hyper_d
                cost = np.sum((hyper[0:band, :] - np.dot(W_hyper[0:band, :], H_hyper))**2)
                if q > 1 and (cost0-cost)/cost < delta_h:
                    if verbose == 'on':
                        print('Initialization of W_hyper converged at the ', q, 'th iteration ')
                    W_hyper = W_hyper_old
                    break
                cost0 = cost
        else:
            # Update H_hyper
            H_hyper_old = H_hyper
            if multi_band > MIN_MS_BANDS:
                H_hyper_n = np.dot(W_hyper.transpose(), hyper)
                H_hyper_d = np.dot(np.dot(W_hyper.transpose(), W_hyper), H_hyper)
                H_hyper = (H_hyper*H_hyper_n)/H_hyper_d
            # Update W_hyper
            W_hyper_old = W_hyper
            W_hyper_n = np.dot(hyper[0:band, :], H_hyper.transpose())
            W_hyper_d = np.dot(np.dot(W_hyper[0:band, :], H_hyper), H_hyper.transpose())
            W_hyper[0:band, :] = (W_hyper[0:band, :]*W_hyper_n)/W_hyper_d
            cost = np.sum((hyper[0:band, :] - np.dot(W_hyper[0:band, :], H_hyper))**2)
            if (cost0-cost)/cost < delta_h:
                if verbose == 'on':
                    print('Optimization of HS unmixing converged at the ', i, 'th iteration ')
                H_hyper = H_hyper_old
                W_hyper = W_hyper_old
                break
            cost0 = cost

    RMSE_h = (cost0/(hyper.shape[1]*band))**0.5
    if verbose == 'on':
        print('    RMSE(Vh) = ', RMSE_h)

    W_multi1 = W_multi.copy()
    H_multi1 = H_multi.copy()

    # initialize W_multi: (multi_band, M)
    W_multi[0:multi_band,:] = np.dot(srf, W_hyper[0:band,:])

    if verbose == 'on':
        print('NMF for Vm (', i_out+2, ')')
    for i in range(I_in):
        if i == 0:
            cost0 = 0
            for q in range(I_in):
                # Update H_multi
                H_multi_old = H_multi
                H_multi_n = np.dot(W_multi.transpose(), multi)
                H_multi_d = np.dot(np.dot(W_multi.transpose(), W_multi), H_multi)
                H_multi = (H_multi*H_multi_n)/H_multi_d
                cost = np.sum((multi[0:multi_band, :] - np.dot(W_multi[0:multi_band, :], H_multi))**2)
                if q > 1 and (cost0-cost)/cost < delta_m:
                    if verbose == 'on':
                        print('Initialization of H_multi converged at the ', q, 'th iteration ')
                    H_multi = H_multi_old
                    break
                cost0 = cost
        else:
            # Update W_multi
            W_multi_old = W_multi
            if multi_band > MIN_MS_BANDS:
                W_multi_n = np.dot(multi[0:multi_band, :], H_multi.transpose())
                W_multi_d = np.dot(np.dot(W_multi[0:multi_band, :], H_multi), H_multi.transpose())
                W_multi[0:multi_band, :] = (W_multi[0:multi_band, :]*W_multi_n)/W_multi_d
            # Update H_multi
            H_multi_old = H_multi
            H_multi_n = np.dot(W_multi.transpose(), multi)
            H_multi_d = np.dot(np.dot(W_multi.transpose(), W_multi), H_multi)
            H_multi = (H_multi*H_multi_n)/H_multi_d
            cost = np.sum((multi[0:multi_band, :] - np.dot(W_multi[0:multi_band, :], H_multi))**2)
            if (cost0-cost)/cost < delta_m:
                if verbose == 'on':
                    print('Optimization of MS unmixing converged at the ', i, 'th iteration ')
                W_multi = W_multi_old
                H_multi = H_multi_old
                break
            cost0 = cost

    RMSE_m = (cost0/(multi.shape[1]*multi_band))**0.5
    if verbose == 'on':
        print('    RMSE(Vm) = ', RMSE_m)

    W_multi2 = W_multi
    H_multi2 = H_multi

    return W_hyper, H_hyper, W_multi1, H_multi1, W_multi2, H_multi2, RMSE_h, RMSE_m

def gaussian_filter2d(shape=(3,3),sigma=1):
    '''
    2D Gaussian filter

    USAGE
        h = gaussian_filter2d(shape,sigma)

    INPUT
        shape : window size (e.g., (3,3))
        sigma : scalar

    OUTPUT
        h
    '''
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x**2 + y**2) / (2.*sigma**2) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def gaussian_down_sample(data,w,mask=0):
    '''
    This function downsamples HS image with a Gaussian point spread function.

    USAGE
          HSI = gaussian_down_sample(data,w,mask)

    INPUT
          data            : input HS image (xdata,ydata,band)
          w               : difference of ground sampling distance (FWHM = w)
          mask            : (optional) Binary mask for processing (xdata,ydata) (0: mask, 1: image)

    OUTPUT
          HSI             : downsampled HS image (xdata/w, ydata/w, band)
    '''

    # masking mode
    if np.isscalar(mask):
        masking = 0
    else:
        masking = 1

    xdata = data.shape[0]
    ydata = data.shape[1]
    band = data.shape[2]
    hx = int(np.floor(xdata/w))
    hy = int(np.floor(ydata/w))
    HSI = np.zeros((hx, hy, band))
    sig = w/2.35482

    if masking == 0: # without mask
        if np.mod(w,2)==0:
            H1 = gaussian_filter2d((w,w),sig).reshape(w,w,1)
            H2 = gaussian_filter2d((w*2,w*2),sig).reshape(w*2,w*2,1)
            for x in range(hx):
                for y in range(hy):
                    if x==0 or x==hx-1 or y==0 or y==hy-1:
                        HSI[x,y,:] = (np.double( data[x*w:(x+1)*w,y*w:(y+1)*w,:] ) * np.tile(H1,(1,1,band))).sum(axis=0).sum(axis=0).reshape(1,1,band)
                    else:
                        HSI[x,y,:] = (np.double( data[x*w-w//2:(x+1)*w+w//2,y*w-w//2:(y+1)*w+w//2,:] ) * np.tile(H2,(1,1,band))).sum(axis=0).sum(axis=0).reshape(1,1,band)
        else:
            H1 = gaussian_filter2d((w,w),sig).reshape(w,w,1)
            H2 = gaussian_filter2d((w*2-1,w*2-1),sig).reshape(w*2-1,w*2-1,1)
            for x in range(hx):
                for y in range(hy):
                    if x==0 or x==hx-1 or y==0 or y==hy-1:
                        HSI[x,y,:] = (np.double( data[x*w:(x+1)*w,y*w:(y+1)*w,:] ) * np.tile(H1,(1,1,band)) ).sum(axis=0).sum(axis=0).reshape(1,1,band)
                    else:
                        HSI[x,y,:] = (np.double( data[x*w-(w-1)//2:(x+1)*w+(w-1)//2,y*w-(w-1)//2:(y+1)*w+(w-1)//2,:] ) * np.tile(H2,(1,1,band))).sum(axis=0).sum(axis=0).reshape(1,1,band)
    else: # with mask
        if np.mod(w,2)==0:
            H1 = gaussian_filter2d((w,w),sig).reshape(w,w,1)
            H2 = gaussian_filter2d((w*2,w*2),sig).reshape(w*2,w*2,1)
            for x in range(hx):
                for y in range(hy):
                    mask_tmp = mask[x*w:(x+1)*w,y*w:(y+1)*w]
                    if mask_tmp.sum() == w**2:
                        if x==0 or x==hx-1 or y==0 or y==hy-1:
                            HSI[x,y,:] = (np.double( data[x*w:(x+1)*w,y*w:(y+1)*w,:] ) * np.tile(H1,(1,1,band))).sum(axis=0).sum(axis=0).reshape(1,1,band)
                        else:
                            HSI[x,y,:] = (np.double( data[x*w-w//2:(x+1)*w+w//2,y*w-w//2:(y+1)*w+w//2,:] ) * np.tile(H2,(1,1,band))).sum(axis=0).sum(axis=0).reshape(1,1,band)
        else:
            H1 = gaussian_filter2d((w,w),sig).reshape(w,w,1)
            H2 = gaussian_filter2d((w*2-1,w*2-1),sig).reshape(w*2-1,w*2-1,1)
            for x in range(hx):
                for y in range(hy):
                    mask_tmp = mask[x*w:(x+1)*w,y*w:(y+1)*w]
                    if mask_tmp.sum() == w**2:
                        if x==0 or x==hx-1 or y==0 or y==hy-1:
                            HSI[x,y,:] = (np.double( data[x*w:(x+1)*w,y*w:(y+1)*w,:] ) * np.tile(H1,(1,1,band)) ).sum(axis=0).sum(axis=0).reshape(1,1,band)
                        else:
                            HSI[x,y,:] = (np.double( data[x*w-(w-1)//2:(x+1)*w+(w-1)//2,y*w-(w-1)//2:(y+1)*w+(w-1)//2,:] ) * np.tile(H2,(1,1,band))).sum(axis=0).sum(axis=0).reshape(1,1,band)

    return HSI

def zoom_nn(data,w):
    '''
    Zoom via nearest neighbor interpolation
    '''
    rows = data.shape[0]
    cols = data.shape[1]
    print(data.shape)
    out = np.tile( np.tile(data.reshape(rows,cols,1),(1,1,w)).reshape(rows,cols*w,1) ,(1,1,w)).transpose(1,0,2).reshape(cols*w,rows*w).transpose()

    return out

def zoom_bi(data,w):
    '''
    Zoom via bilinear interpolation
    '''
    rows = data.shape[0]
    cols = data.shape[1]
    # index
    r = np.tile(((2*np.r_[0:rows*w]+1)/(2*w)-0.5).reshape(rows*w,1),(1,cols*w))
    c = np.tile((2*np.r_[0:cols*w]+1)/(2*w)-0.5,(rows*w,1))
    r[r<0] = 0
    r[r>rows-1] = rows-1
    c[c<0] = 0
    c[c>cols-1] = cols-1
    w4 = (np.floor(r)+1-r)*(np.floor(c)+1-c)
    w3 = (np.floor(r)+1-r)*(c-np.floor(c))
    w2 = (r-np.floor(r))*(np.floor(c)+1-c)
    w1 = (r-np.floor(r))*(c-np.floor(c))
    data = np.hstack((np.vstack((data,np.zeros((1,cols)))),np.zeros((rows+1,1))))
    out = w4*data[np.floor(r).astype(int),np.floor(c).astype(int)]+w3*data[np.floor(r).astype(int),np.floor(c).astype(int)+1]+w2*data[np.floor(r).astype(int)+1,np.floor(c).astype(int)]+w1*data[np.floor(r).astype(int)+1,np.floor(c).astype(int)+1]

    return out

def lsqnonneg(y,A):
    '''
    Nonnegative least squares via the active set method

    This function solves the following optimization

        min |y-Ax|^2
        s.t. x>=0

    USAGE
        x = lsqnonneg(y,A)

    INPUT
        y  : observation (m,1)
        A  : mixing matrix (m,n)

    OUTPUT
        x  : coefficients (n,1)
    '''

    t = 10*2.2204e-16*np.max(np.sum(np.abs(A),axis=0))*max([A.shape[0], A.shape[1]])

    m = y.shape[0]
    n = A.shape[1]

    # initialize
    x = np.zeros((n,1))
    s = x.copy()
    P = np.zeros((n,1))
    R = np.ones((n,1))
    w = np.dot(A.transpose() , (y - np.dot(A,x)))

    # main loop
    c = 0
    while R.sum() > 0 and w.max() > t:
        if c > 0:
            j_pre = j
        j = np.nonzero(w==w.max())
        if c > 0:
            if j == j_pre:
                break
        c = c+1

        P[j[0]] = 1
        R[j[0]] = 0
        Ap = A[:,np.nonzero(P==1)[0]]
        sp = np.dot( np.linalg.inv(np.dot(Ap.transpose(),Ap)) , np.dot(Ap.transpose(),y) )
        s[np.nonzero(P==1)] = sp.reshape(1,len(sp))[0,:]
        while s[np.nonzero(P==1)].min() <= 0:
            if sum((s<=0)*((x-s)!=0)) != 0:
                alpha = ( x[(s<=0)*((x-s)!=0)] / (x[(s<=0)*((x-s)!=0)]-s[(s<=0)*((x-s)!=0)]) ).min()
                x = x + alpha*(s-x)
                R[np.nonzero(x==0)] = 1
                P[np.nonzero(x==0)] = 0
                Ap = A[:,np.nonzero(P==1)[0]]
                sp = np.dot( np.linalg.inv(np.dot(Ap.transpose(),Ap)) , np.dot(Ap.transpose(),y) )
                s[np.nonzero(P==1)] = sp.reshape(1,len(sp))[0]
                s[np.nonzero(R==1)] = 0
            else:
                break
        x = s.copy()
        w = np.dot(A.transpose() , (y - np.dot(A,x)))

    return x

def nls_su(Y,A):
    '''
    Nonnegative least squares for spectral unmixing

    This function solves the following optimization

        min |Y-AX|_F^2
        s.t. X>=0

    USAGE
        X = nls_su(Y,A)

    INPUT
        Y  : observation (m,p)
        A  : mixing matrix (m,n)

    OUTPUT
        X  : coefficients (n,p)
    '''
    n = A.shape[1]
    p = Y.shape[1]
    m = Y.shape[0]
    X = np.zeros((p,n))
    for i in range(p):
        y = Y[:,i].reshape(m,1).copy()
        x = lsqnonneg(y,A)
        X[i,:] = x.transpose().copy()
    print(n, p)

    return X.transpose()

def estR(HS,MS,mask=0):
    '''
    Estimation of relative spectral response functions (SRFs)
    via the nonnegative least squares method

    USAGE
        R = estR(HS,MS,mask)

    INPUT
        HS  : Low-spatial-resolution HS image (rows2,cols2,bands2)
        MS  : MS image (rows1,cols1,bands1)
        mask: (optional) Binary mask for processing (rows2,cols2) (mainly
              for real data)

    OUTPUT
        R   : Relative SRFs
              without mask (bands1,bands2)
              with mask    (bands1,bands2+1) (consider offset)
    '''

    rows1 = MS.shape[0]
    cols1 = MS.shape[1]
    bands1 = MS.shape[2]
    rows2 = HS.shape[0]
    cols2 = HS.shape[1]
    bands2 = HS.shape[2]

    # masking mode
    if np.isscalar(mask):
        masking = 0
        mask = np.ones((rows2,cols2))
    else:
        masking = 1

    HS = np.hstack((HS.reshape(rows2*cols2,bands2), mask.reshape(rows2*cols2,1) )).reshape(rows2,cols2,bands2+1)
    bands2 = HS.shape[2]

    R = np.zeros((bands1,bands2))

    # downgrade spatial resolution
    w = int(rows1/rows2)
    mask2 = zoom_nn(mask,w)

    Y = gaussian_down_sample(MS,w,mask2).reshape(rows2*cols2,bands1)

    A = HS.reshape(rows2*cols2,bands2).copy()

    if masking == 1:
        Y = Y[mask.reshape(rows2*cols2)==1,:]
        A = A[mask.reshape(rows2*cols2)==1,:]

    # solve nonnegative least squares problems
    for b in range(bands1):
        y = Y[:,b].reshape(Y.shape[0],1).copy()
        r = lsqnonneg(y,A)
        R[b,:] = r.transpose().copy()

    return R

def vca(R,p):
    '''
    Vertex Component Analysis (VCA)

    USAGE
        U, indices = vca( R, p )

    INPUT
        R  : Hyperspectral data (bands,pixels)
        p  : Number of endmembers

    OUTPUT
        U  : Matrix of endmembers (bands,p)
        indices : Indices of endmembers in R

    REFERENCE
    J. M. P. Nascimento and J. M. B. Dias, "Vertex component analysis: A
    fast algorithm to unmix hyperspectral data," IEEE Transactions on
    Geoscience and Remote Sensing, vol. 43, no. 4, pp. 898 - 910, Apr. 2005.
    '''

    N = R.shape[1] # pixels
    L = R.shape[0] # bands

    # Estimate SNR
    r_m = R.mean(axis=1).reshape(L,1)
    R_o = R - np.tile(r_m, (1, N))
    U, S, V = np.linalg.svd(np.dot(R_o,R_o.T) / N)
    Ud = U[:,:p] # computes the p-projection matrix
    x_p = np.dot(Ud.T, R_o)
    P_y = (R**2).sum() / N
    P_x = (x_p**2).sum() / N + np.dot(r_m.T, r_m)
    SNR = np.abs(10*np.log10( (P_x - (p/L)*P_y) / (P_y - P_x) ))

    # Determine which projection to use.
    SNRth = 15 + 10*np.log(p) + 8
    #SNRth = 15 + 10*log(p) # threshold proposed in the original paper
    if SNR > SNRth:
        d = p
        Ud, Sd, Vd = np.linalg.svd(np.dot(R,R.T)/N)
        Ud = U[:,:d]
        X = np.dot(Ud.T,R)
        u = X.mean(axis=1).reshape(X.shape[0],1)
        Y = X / np.tile( ( X * np.tile(u,(1, N)) ).sum(axis = 0) ,(d, 1) )
    else:
        d = p-1
        r_m = (R.T).mean(axis=0).reshape((R.T).shape[1],1)
        R_o = R - np.tile(r_m, (1, N))
        Ud, Sd, Vd = np.linalg.svd(np.dot(R_o,R_o.T)/N)
        Ud = U[:,:d]
        X = np.dot(Ud.T, R_o)
        c = np.sqrt((X**2).sum(axis = 0).max())
        c = np.tile(c, (1, N))
        Y = np.vstack( (X, c) )

    e_u = np.zeros((p, 1))
    e_u[p-1,0] = 1
    A = np.zeros((p, p))
    A[:,0] = e_u[:,0]

    I = np.eye(p)
    k = np.zeros((N, 1))

    indices = []
    for i in range(p):
        w = np.random.rand(p,1)
        f = np.dot((I-np.dot(A,np.linalg.pinv(A))), w)
        f = f / np.linalg.norm(f)
        v = np.dot(f.T,Y)
        k = np.abs(v).argmax()
        A[:,i] = Y[:,k]
        indices.append(k)

    if SNR > SNRth:
        U = np.dot(Ud,X[:,indices])
    else:
        U = np.dot(Ud,X[:,indices]) + np.tile(r_m, (1, p))

    return U, indices

def vd(data,alpha=10**(-3)):
    '''
    Virtual dimensionality

    USAGE
        out = vd(data,alpha)

    INPUT
        data : HSI data (bands,pizels)
        alpha: False alarm rate

    OUTPUT
        out  : Number of spectrally distinct signal sources in data

    REFERENCE
    J. Harsanyi, W. Farrand, and C.-I Chang, "Determining the number and
    identity of spectral endmembers: An integrated approach using
    Neyman-Pearson eigenthresholding and iterative constrained RMS error
    minimization," in Proc. 9th Thematic Conf. Geologic Remote Sensing,
    Feb. 1993.
    Chang, C.-I. and Du, Q., "Estimation of number of spectrally distinct
    signal sources in hyperspectral imagery," IEEE Transactions on Geoscience
    and Remote Sensing, vol. 42, pp. 608-619, 2004.
    '''
    data = np.double(data)
    N = data.shape[1] # pixels
    L = data.shape[0] # bands

    R = np.dot(data, data.T)/N
    K = np.cov(data)

    D_r, V_r = np.linalg.eig(R)
    D_k, V_k = np.linalg.eig(K)

    e_r = np.sort(D_r)[::-1]
    e_k = np.sort(D_k)[::-1]

    diff = e_r - e_k
    variance = (2*(e_r**2+e_k**2)/N)**0.5

    tau = -ppf(alpha,np.zeros(L),variance)

    out = sum(diff > tau)

    return out

def PSNR(ref,tar,mask=0):
    '''
    Peak signal to noise ratio (PSNR)

    USAGE
        psnr_all, psnr_mean = PSNR(ref,tar)

    INPUT
        ref : reference HS data (rows,cols,bands)
        tar : target HS data (rows,cols,bands)
        mask: (optional) Binary mask for processing  (rows,cols) (0: mask, 1: image)

    OUTPUT
        psnr_all  : PSNR (bands)
        psnr_mean : average PSNR (scalar)
    '''
    rows = ref.shape[0]
    cols = ref.shape[1]
    bands = ref.shape[2]

    # masking mode
    if np.isscalar(mask):
        mask = np.ones((rows,cols))

    ref = ref.reshape(rows*cols,bands)
    tar = tar.reshape(rows*cols,bands)
    mask = mask.reshape(rows*cols)
    msr = ((ref[mask==1,:]-tar[mask==1,:])**2).mean(axis=0)
    max2 = ref.max(axis=0)**2

    psnr_all = 10*np.log10(max2/msr)
    psnr_mean = psnr_all.mean()

    return psnr_all, psnr_mean

def SAM(ref,tar,mask=0):
    '''
    Spectral angle mapper (SAM)

    USAGE
        sam_mean, map = SAM(ref,tar)

    INPUT
        ref : reference HS data (rows,cols,bands)
        tar : target HS data (rows,cols,bands)
        mask: (optional) Binary mask for processing  (rows,cols) (0: mask, 1: image)

    OUTPUT
        sam_mean : average value of SAM (scalar in degree)
        map      : 2-D map (in degree)
    '''
    rows = tar.shape[0]
    cols = tar.shape[1]
    bands = tar.shape[2]

    # masking mode
    if np.isscalar(mask):
        masking = 0
        mask = np.ones(rows*cols)
    else:
        masking = 1
        mask = mask.reshape(rows*cols)

    prod_scal = (ref*tar).sum(axis=2)
    norm_orig = (ref*ref).sum(axis=2)
    norm_fusa = (tar*tar).sum(axis=2)
    prod_norm = np.sqrt(norm_orig*norm_fusa)
    prod_map = prod_norm
    prod_map[prod_map==0] = 2.2204e-16
    map = np.real(np.arccos(prod_scal/prod_map))*180/np.pi
    prod_scal = prod_scal.reshape(rows*cols)
    prod_norm = prod_norm.reshape(rows*cols)
    sam_mean = np.real(np.arccos(prod_scal[(prod_norm!=0)*(mask==1)]/prod_norm[(prod_norm!=0)*(mask==1)]).sum()/((prod_norm!=0)*(mask==1)).sum())*180/np.pi

    return sam_mean, map

def ppf(p,mu=0,sigma=1):
    '''
    Percent point function (inverse of cdf)
    for the normal distribution at p

    USAGE
        out = ppf(p,mu,sigma)

    INPUT
        p     : lower tail probability
        mu    : mean (n)
        sigma : standard deviation (n)

    OUTPUT
        out   : quantile corresponding to the lower tail probability p (n)
    '''
    n = mu.shape[0] # number of elements
    out = np.zeros((n))
    for i in range(n):
        #print sigma[i]
        out[i] = 2**0.5*sigma[i]*erfinv(2*p-1)+mu[i]

    return out