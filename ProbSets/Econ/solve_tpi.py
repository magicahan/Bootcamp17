#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 12:09:48 2017

@author: luxihan
"""
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
from OG_ss import *

def euler_tpi(bvec, *args):
    alpha, sigma, delta, beta, A, nvec, b_init, w, r = args
    bvec_new = np.append([b_init], bvec)
#    print(w.shape, bvec_new.shape)
    params = (alpha, sigma, delta, beta, A)
    cvec = get_cvec(w, r, bvec_new, nvec, params)
    error = cvec[:-1]**(-sigma) - beta * (1 + r[1:]) * cvec[1:] ** (-sigma)
    return error


def path_life(w, r, b_init, bvec_guess, params):
    alpha, sigma, delta, beta, A, nvec, tol = params
    nvec = nvec[-len(bvec_guess) - 1:]
    eul_args = (alpha, sigma, delta, beta, A, nvec, b_init, w, r)
    b_path = opt.fsolve(euler_tpi, x0 = bvec_guess, args=(eul_args),
                          xtol=tol)
    c_path = get_cvec(w, r, np.append([b_init,], b_path), nvec, params[:5])
    return b_path, c_path

def get_path_mat(K, b_vec_init, params):
    S, T, alpha, sigma, delta, beta, A, nvec, tol = params
    L = nvec.sum()
    b_path_mat = np.zeros((S - 1, T + S))
    c_path_mat = np.zeros((S, T + S))
    b_path_mat[:, 0] = b_vec_init
    w_path = get_w(K, L, params[2:-2])
    r_path = get_r(K, L, params[2:-2])
    #compute the consumption of the last cohort
    c_init_old = get_cvec(w_path[0], r_path[0], b_vec_init[-1], nvec[-1], params[2 : -2])
    c_path_mat[0, -1] = c_init_old
    for t in range(S - 2):
        b_init = b_vec_init[-(t + 2)]
        bvec_guess = b_vec_init[-(t + 1) : ]
        b_path, c_path = path_life(w_path[:t + 2],\
                        r_path[:t + 2], b_init, bvec_guess, params[2:])
        b_path_mat[-(t + 1):, 1 : t + 2] += np.diag(b_path)
        c_path_mat[- (t + 2):, : t + 2] += np.diag(c_path)
        
    for t in range(T + 1):
        b_init = 0
        bvec_guess = np.diag(b_path_mat[:, t : t + S - 1])
        b_path, c_path = path_life(w_path[t : t + S], r_path[t : t + S], \
                                   b_init, bvec_guess, params[2 :])
        b_path_mat[:, t + 1 : t + S] += np.diag(b_path)
        c_path_mat[:, t : t + S] += np.diag(c_path)
        
        
    return b_path_mat, c_path_mat

def solve_tpi(b_vec_init, params, spec, graph):
    S, T, alpha, sigma, delta, beta, A, nvec, tol, eta = params
    L = nvec.sum()
    K_init = b_vec_init.sum()
    ss_params = (alpha, sigma, delta, beta, A, nvec, tol)
    ss_dict = get_ss(b_vec_init, ss_params)
    K_ss = ss_dict['K_ss']
    kpath_now = get_path_k(K_init, K_ss, T + S, spec)
    kpath_prev = np.zeros(T + S)
    while abs((kpath_now ** 2 - kpath_prev ** 2).sum()) > tol:
        if kpath_prev.sum() != 0:
            kpath_now = eta * kpath_now + (1 - eta) * kpath_prev
        b_path_mat, c_path_mat = get_path_mat(kpath_now, b_vec_init, params[:-1])
        kpath_prev = kpath_now
        kpath_now = kpath_prev.copy()
        kpath_now[:b_path_mat.shape[1]] = b_path_mat.sum(axis = 0)
        
    #generate the path of consumption, output, rental rate, wage rate
    k_path = b_path_mat.sum(axis = 0)
    w_path = get_w(k_path, L, params[2 : -3])
    r_path = get_r(k_path, L, params[2 : -3])
    c_path = c_path_mat.sum(axis = 0)
    y_path = get_y(k_path, L, params[2 : -3])
    path_dict = {'k_path' : k_path, 'y_path' : y_path, 'w_path' : w_path, 'c_path': c_path,\
                 'r_path' : r_path, 'c_path_mat' : c_path_mat, 'b_path_mat':b_path_mat}
    
    #generate the graph
    if graph:
        plt.plot(k_path[:T],\
                 marker = 'D', label = 'Capital Path')
        plt.ylabel('Capital K')
        plt.xlabel('Period T')
        plt.suptitle('Capital Path')
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.legend(loc='upper right')
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, "capital")
        plt.savefig(output_path,bbox_inches='tight')
        plt.show()
    
    return path_dict
    
    
def get_path_k(K_init, K_ss, T, spec):
    if spec == 'linear':
        kpath = np.linspace(K_init, K_ss, T)
    elif spec == 'quadratic':
        cc = K_init
        bb = 2 * (K_ss - K_init) / (T - 1)
        aa = (K_init - K_ss) / ((T - 1) ** 2)
        kpath = aa * (np.arange(0, T) ** 2) + (bb * np.arange(0, T)) + cc
    return kpath

if __name__ == '__main__':
    #firm side parameter
    alpha = 0.35
    A = 1
    
    #household parameter
    beta = 0.442
    delta = 0.6415
    sigma = 3
    nvec = np.array([1, 1, 0.2])
    bvec_guess = np.array([0.2, 0.2])
    
    tol = 1e-9
    
    S = 3
    T = 40
    eta = 0.4
    
    ss_params = (alpha, sigma, delta, beta, A, nvec, tol)
    params = (S, T, alpha, sigma, delta, beta, A, nvec, tol, eta)
    rv_dict = get_ss(bvec_guess, ss_params)
    b_ss = rv_dict['b_ss']
    b_vec_init = np.array([0.8, 1.1]) * b_ss
    b_path_mat = solve_tpi(b_vec_init, params, 'linear', True)
    
        
    
    