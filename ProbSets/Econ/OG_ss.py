#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 17:02:27 2017

@author: luxihan
"""
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os


def get_l(nvec, params):
    alpha, sigma, delta, beta, A = params
    return nvec.sum(axis = 0)


def get_k(bvec, params):
    alpha, sigma, delta, beta, A = params
    return bvec.sum(axis = 0)

def get_w(K, L, params):
    alpha, sigma, delta, beta, A = params
    w = (1 - alpha) * A * (K/L)**alpha
    return w

def get_r(K, L, params):
    alpha, sigma, delta, beta, A = params
    r = alpha * A * (L/K)**(1 - alpha) - delta
    return r

def get_y(K, L, params):
    alpha, sigma, delta, beta, A = params
    y = A * K**alpha * L**(1 - alpha)
    return y
    

def get_cvec(w, r, bvec, nvec, params):
    alpha, sigma, delta, beta, A = params
    if bvec.ndim <= 1:
        bvec_new = np.append(bvec, 0)
        cvec = w * nvec - bvec_new[1:] + (1 + r) * (bvec_new[:-1])
        return cvec
    else:
        cvec = w * nvec - bvec[1:] + (1 + r) * (bvec[:-1])
        return cvec
    
def euler_ss(bvec, *args):
    alpha, sigma, delta, beta, A, nvec= args
    params = np.array([alpha, sigma, delta, beta, A])
    K = bvec.sum()
    L = nvec.sum()
    r = get_r(K, L, params)
    w = get_w(K, L, params)
    print(r)
    cvec = get_cvec(w, r, np.append([0], bvec), nvec, params)
    error = cvec[:-1]**(-sigma) - beta * (1 + r) * cvec[1:] ** (-sigma)
    return error

def get_ss(bvec_guess, params):
    alpha, sigma, delta, beta, A, nvec, tol = params
    eul_args = (alpha, sigma, delta, beta, A, nvec)
    params = alpha, sigma, delta, beta, A
    b_ss = opt.fsolve(euler_ss, x0 = bvec_guess, args=(eul_args),
                          xtol=tol)
    K_ss = b_ss.sum()
    L_ss = nvec.sum()
    w_ss = get_w(K_ss, L_ss, params)
    r_ss = get_r(K_ss, L_ss, params)
    c_ss = get_cvec(w_ss, r_ss, np.append([0], b_ss), nvec, params)
    y_ss = get_y(K_ss, L_ss, params)
    Res_err = y_ss - c_ss.sum() - delta * K_ss
    Euler_err = euler_ss(b_ss, alpha, sigma, delta, beta, A, nvec)
    rv_result = {'b_ss': b_ss, 'c_ss': c_ss, 'r_ss': r_ss, 'w_ss': w_ss, \
                 'y_ss': y_ss, 'K_ss': K_ss, 'Res_err': Res_err, \
                 'Eul_err': Euler_err}
    return rv_result

if __name__ == '__main__':
    #firm side parameter
    alpha = 0.35
    A = 1
    
    #household parameter
    beta1 = 0.442
    beta2 = 0.3585
    delta = 0.6415
    sigma = 3
    nvec = np.array([1, 1, 0.2])
    bvec_guess = np.array([0.2, 0.2])
    
    tol = 1e-12
    
    params1 = (alpha, sigma, delta, beta1, A, nvec, tol)
    rv_dict1 = get_ss(bvec_guess, params1)
    
    params2 = (alpha, sigma, delta, beta2, A, nvec, tol)
    rv_dict2 = get_ss(bvec_guess, params2)
    
    for var in rv_dict.keys():
        print('Beta: {}; {} : {}'.format(beta1, var, rv_dict1[var]))
        print('Beta: {}; {} : {}'.format(beta2, var, rv_dict2[var]))
        print()
    