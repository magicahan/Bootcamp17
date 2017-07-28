#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 00:47:17 2017

@author: luxihan
"""

from LinApp_FindSS import LinApp_FindSS
from LinApp_Deriv import LinApp_Deriv
from LinApp_Solve import LinApp_Solve
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
import os
np.random.seed(12345)

def brock_linear(kgrid, zgrid, kbar, params):
    alpha, rho = params
    F = (alpha * kbar ** (alpha - 1))/(kbar**alpha - kbar)
    G = - (alpha * kbar **(alpha - 1) * (alpha + kbar**(alpha - 1)))\
          /(kbar ** alpha - kbar)
    H = alpha**2 * kbar**(2*(alpha - 1))/(kbar**alpha - kbar)
    L = - (alpha * kbar**(2*alpha - 1))/(kbar**alpha - kbar)
    M = alpha**2 * kbar**(2*(alpha - 1))/(kbar**alpha - kbar)
    
    P1 = (-G + np.sqrt(G**2 - 4*F*H))/(2*F)
    P2 = (-G - np.sqrt(G**2 - 4*F*H))/(2*F)
    
    if P1 > 0 and P1 < 1:
        P = P1
    else:
        P = P2
    Q = - (L * rho + M)/(F * rho + F*P + G)
    
    k_exgrid, z_exgrid = np.meshgrid(kgrid, zgrid)
    knext = kbar + P * (k_exgrid - kbar) + Q * z_exgrid
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(k_exgrid, z_exgrid, knext)
    plt.show()
    
def brock_loglinear(kgrid, zgrid, kbar, params):
    alpha, rho = params
    F = (alpha * kbar ** (alpha - 1))/(kbar**alpha - kbar) * kbar
    G = - (alpha * kbar **(alpha - 1) * (alpha + kbar**(alpha - 1)))\
          /(kbar ** alpha - kbar) * kbar
    H = alpha**2 * kbar**(2*(alpha - 1))/(kbar**alpha - kbar) * kbar
    L = - (alpha * kbar**(2*alpha - 1))/(kbar**alpha - kbar)
    M = alpha**2 * kbar**(2*(alpha - 1))/(kbar**alpha - kbar)
    
    P1 = (-G + np.sqrt(G**2 - 4*F*H))/(2*F)
    P2 = (-G - np.sqrt(G**2 - 4*F*H))/(2*F)
    
    if P1 > 0 and P1 < 1:
        P = P1
    else:
        P = P2
    Q = - (L * rho + M)/(F * rho + F*P + G)
    
    k_exgrid, z_exgrid = np.meshgrid(kgrid, zgrid)
    k_exgrid = (k_exgrid - kbar)/kbar
    knext = kbar + P * (k_exgrid - kbar) + Q * z_exgrid
    k_exgrid = k_exgrid * kbar + kbar
    knext = knext * kbar + kbar
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(k_exgrid, z_exgrid, knext)
    plt.show()
    
def model_tax(Xp, X, Z, params):
    gamma, eps, beta, alpha, a, delta, zbar, rho, tau = params
    kp, lpp = Xp
    k, lp = X
    z = Z
    r = alpha * k ** (alpha - 1) * (np.exp(z) * lp) ** (1 - alpha)
    w = (1 - alpha) * (k / lp)**(alpha) * (np.exp(z))**(1 - alpha)
    T = tau * (w * lp + (r - delta) * k)
    c = (1 - tau) * (w * lp + (r - delta) * k) + k + T - kp 
    i = kp - (1 - delta) * k
    return r, w, T, c, i

def model_fulltax(theta0, params):
    gamma, eps, beta, alpha, a, delta, zbar, rho, tau = params
    kpp, lppp, kp, lpp, k, lp, Zp, Z = theta0
    Xpp = np.array([kpp, lppp])
    Xp = np.array([kp, lpp])
    X = np.array([k, lp])
#    Xpp, Xp, X, Zp, Z = theta0
#    kpp, lp = Xp
    
    r, w, T, c, i = model_tax(Xp, X, Z, params)
    rp, wp, Tp, cp, ip = model_tax(Xpp, Xp, Zp, params)
    
    E1 = beta * c**gamma/cp**gamma *(1 + rp - delta) * (1 - tau) - 1
    E2 = a * cp**gamma/(wp * (1-tau)*(1 - lpp)**(eps)) - 1
    return np.array([E1, E2])

def find_ss(xy_init, zbar, params):
    gamma, eps, beta, alpha, a, delta, zbar, rho, tau = params
    XYbar = LinApp_FindSS(model_fulltax, params, guessXY, zbar, nx, ny)
    kbar, lbar = XYbar
    rbar, wbar, Tbar, cbar, ibar = model_tax(XYbar, XYbar, zbar, params)
    return np.array([kbar, lbar, rbar, wbar, Tbar, cbar, ibar])
    

def find_der(xy_init, zbar, params):
    gamma, eps, beta, alpha, a, delta, zbar, rho, tau = params
    ss_vals = find_ss(xy_init, zbar, params)
    der = np.empty((ss_vals.shape[0], params.shape[0]))
    
    for i in range(len(params)):
        if params[i] != 0:
            params1 = params.copy()
            params1[i] += params1[i] * 1e-3
            params2 = params.copy()
            params2[i] -= params2[i] * 1e-3
            dd = 2 * params[i] * 1e-3
        else:
            params1 = params.copy()
            params1[i] += 1e-6
            params2 = params.copy()
            params2[i] -= 1e-6
            dd = 2 * 1e-6
        val1 = find_ss(xy_init, zbar, params1)
        val2 = find_ss(xy_init, zbar, params2)
        der[:, i] = (val1 - val2) / dd
    return der

def simulate(PP, QQ, params, sigma):
    gamma, eps, beta, alpha, a, delta, zbar, rho, tau = params
    error_mat = np.random.normal(loc = 0, scale = sigma**0.5, size = (10000, 250))
    kbar, lbar, rbar, wbar, Tbar, cbar, ibar = find_ss(np.array([0.2, 0.5]), 0, params)
    ybar = kbar ** alpha * lbar ** (1 - alpha)
    ymat = np.empty((10000, 250))
    cmat = np.empty((10000, 250))
    imat = np.empty((10000, 250))
    lmat = np.empty((10000, 250))
    x_prev = np.zeros((2, 10000))
#    x_prev[0, :] *= kbar
#    x_prev[1, :] *= lbar
    z_prev = np.zeros(10000) 
    z_now = rho * z_prev + error_mat[:, 0]
    z_now = z_now.reshape((1, 10000))
    x_now = PP @ x_prev + QQ @ z_now
    xybar = np.array((kbar, lbar)).reshape(2, 1)
    for i in range(249):
        x_prev = x_now
        z_prev = z_now
        z_now = rho * z_prev + error_mat[:, i + 1]
        x_now = PP @ x_prev + QQ @ z_now
        r, w, T, c, inv = model_tax(x_now + xybar, x_prev + xybar, \
                                  z_prev, params)
        ymat[:, i + 1] = (x_prev[0, :] + kbar) ** alpha * ((x_prev[1, :] + lbar) \
            * np.exp(z_prev))**(1 - alpha) - ybar
        cmat[:, i + 1] = (c - cbar).flatten()
        imat[:, i + 1] = (inv - ibar).flatten()
        lmat[:, i + 1] = (x_prev[1, :]).flatten()
    return ymat, cmat, imat, lmat

def plot_series(PP, QQ, params, sigma):
    ymat, cmat, imat, lmat = simulate(PP, QQ, params, sigma)
    kbar, lbar, rbar, wbar, Tbar, cbar, ibar = find_ss(np.array([0.2, 0.5]), 0, params)
    ybar = kbar ** alpha * lbar ** (1 - alpha)
    #for GDP
    ymat_avg = ymat.mean(axis = 0) + ybar
    std = np.apply_along_axis(np.std, 0, ymat)/(10000**0.5)
    ymat95 = ymat_avg + 1.96 * std
    ymat5 =  ymat_avg - 1.96 * std
    plt.plot(ymat_avg)
    plt.plot(ymat95, '--')
    plt.plot(ymat5, '--')
    plt.title('GDP')
    plt.show()
    
    #for consumption
    cmat_avg = cmat.mean(axis = 0) + cbar
    std = np.apply_along_axis(np.std, 0, cmat)/(10000**0.5)
    cmat95 = cmat_avg + 1.96 * std
    cmat5 =  cmat_avg - 1.96 * std
    plt.plot(cmat_avg)
    plt.plot(cmat95, '--')
    plt.plot(cmat5, '--')
    plt.title('Consumption')
    plt.show()
    
    #for invsetment
    imat_avg = imat.mean(axis = 0) + ibar
    std = np.apply_along_axis(np.std, 0, imat)/(10000**0.5)
    imat95 = imat_avg + 1.96 * std
    imat5 =  imat_avg - 1.96 * std
    plt.plot(imat_avg)
    plt.plot(imat95, '--')
    plt.plot(imat5, '--')
    plt.title('Investment')
    plt.show()
    
    #for labor input
    lmat_avg = lmat.mean(axis = 0) + lbar
    std = np.apply_along_axis(np.std, 0, lmat)/(10000**0.5)
    print(std)
    lmat95 = lmat_avg + 1.96 * std
    lmat5 =  lmat_avg - 1.96 * std
    plt.plot(lmat_avg)
    plt.plot(lmat95, '--')
    plt.plot(lmat5, '--')
    plt.title('Labor Input')
    plt.show()
    
    val_dict = {'consumption': cmat_avg, 'gdp': ymat_avg, 'invest':imat_avg, \
                'labor':lmat_avg}
    stats_dict = {}
    for key in val_dict.keys():
        mat = val_dict[key]
        temp_dict = {}
        temp_dict['mean'] = mat.mean()
        temp_dict['volatility'] = mat.std()
        temp_dict['variation'] = mat.std()/mat.mean()
        temp_dict['rela_volatility'] = mat.std()/val_dict['gdp'].std()
        temp_dict['cyclicality'] = np.corrcoef(mat, val_dict['gdp'])[0, 1]
        stats_dict[key] = temp_dict
    return ymat_avg, cmat_avg, imat_avg, lmat_avg, stats_dict


def irf(PP, QQ, params, sigma, period):
    gamma, eps, beta, alpha, a, delta, zbar, rho, tau = params
    xy_prev = np.array([0, 0])
    karr = [0]
    larr = [0]
    yarr = [0]
    carr = [0]
    iarr = [0]
    
    zarr = [sigma]
    
    z_prev = sigma
    kbar, lbar, rbar, wbar, Tbar, cbar, ibar = \
        find_ss(np.array([0.2, 0.5]), 0, params)
    
    xybar = np.array([kbar, lbar]).reshape(2, 1)
    
    for i in range(period):
        xy_prev = np.array([karr[i], larr[i]]).reshape(2,1)
        xy_next = (PP @ xy_prev).reshape(2, 1) + QQ * zarr[i]
        k_next, l_next = xy_next
        z = zarr[i]
        karr.append(k_next)
        larr.append(l_next)
        r, w, T, c, inv = model_tax(xy_next + xybar, xy_prev + xybar, \
                                  z, params)
        y = (xy_prev[0] + kbar) ** alpha * ((xy_prev[1] + lbar) \
            * np.exp(z))**(1 - alpha) - ybar
        zarr.append(rho * z)
        yarr.append(y)
        carr.append(c - cbar)
        iarr.append(inv - ibar)
        
    plt.subplot(2, 2, 1)
    plt.plot(yarr)
    plt.title('GDP')
    
    plt.subplot(2, 2, 2)
    plt.plot(carr)
    plt.title('Consumption')
    
    plt.subplot(2, 2, 3)
    plt.plot(iarr)
    plt.title('Investment')
    
    plt.subplot(2, 2, 4)
    plt.plot(karr)
    plt.title('Capital')
    plt.tight_layout()
    plt.show()
    
    return karr, larr, yarr, carr, iarr
        
    
    
    
    
    
    
if __name__=='__main__':
    alpha = 0.35
    beta = 0.98
    rho = 0.95
    sigma = 0.02
    params = np.array([alpha, rho])
    kbar = (alpha * beta)**(1/(1-alpha))
    
    kgrid = np.linspace(0.5*kbar, 1.5*kbar, 100)
    zgrid = np.linspace(-5 * sigma, 5 * sigma, 100)
    
    brock_linear(kgrid, zgrid, kbar, params)
    brock_loglinear(kgrid, zgrid, kbar, params)
    
    
    zbar = 0
    nx = 2
    ny = 0
    nz = 1
    gamma = 2.5
    eps = 1.5
    beta = .98
    alpha = .40
    a = .5
    delta = .10
    rho = .9
    tau = .05
    params = np.array([gamma, eps, beta, alpha, a, delta, zbar, rho, tau])
    guessXY = np.array([0.2, 0.5])
    logX = 0
    Sylv = 0
    
    XYbar = LinApp_FindSS(model_fulltax, params, guessXY, zbar, nx, ny)
    kbar, lbar = XYbar
    rbar, wbar, Tbar, cbar, ibar = model_tax(XYbar, XYbar, zbar, params)
    ybar = kbar ** alpha * lbar ** (1 - alpha)
    print ('Tbar: ', Tbar)
    print ('wbar: ', wbar)
    print ('rbar: ', rbar)
    print ('cbar: ', cbar)
    print ('kbar: ', kbar)
    print ('lbar: ', lbar)
    
    jacob = find_der(guessXY, zbar, params)
    
    theta0 = np.concatenate((XYbar, XYbar, XYbar, np.array([zbar]), np.array([zbar])))
    [AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL
     , MM, WW, TT] = \
        LinApp_Deriv(model_fulltax, params, theta0, nx, ny, nz, logX)
    print('FF: ', FF)
    print('GG: ', GG)
    print('HH: ', HH)
    print('LL: ', LL)
    print('MM: ', MM)
    
    # set value for NN    
    NN = rho
        
    # find the policy and jump function coefficients
    PP, QQ, UU, RR, SS, VV = \
        LinApp_Solve(AA,BB,CC,DD,FF,GG,HH,JJ,KK,LL,MM,WW,TT,NN,zbar,Sylv)
        
    sigma = 0.0004
#    params = np.array([gamma, eps, beta, alpha, a, delta, zbar, rho, tau, sigma])
    ymat, cmat, imat, lmat = simulate(PP, QQ, params, sigma)
    
    ymat_avg, cmat_avg, imat_avg, lmat_avg, stats_dict = plot_series(PP, QQ, params, sigma)
    
    period = 40
    karr, larr, yarr, carr, iarr =  irf(PP, QQ, params, sigma, period)