#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 17:57:33 2017

@author: luxihan
"""

import numpy as np
import scipy
import scipy.optimize as opt
import scipy.interpolate as interp

#DSGE

#Problem 5
def eq5(k, *args):
    params = args[0]
    gamma, beta, alpha, delta, z, tau = params
    w = (1 - alpha) * (k)**alpha
    r = alpha * (1/k)**(1 - alpha)
    c = (w + r * k) * (1 - tau) - delta * k
    return beta * ((1 - tau)*(r - delta) + 1) - 1

#Problem 6
def eq6(vars, *args):
    params = args[0]
    gamma, beta, alpha, delta, z, tau, a, eps = params
    k, l = vars
    w = (1 - alpha) * (k/l)**alpha
    r = alpha * (l/k)**(1 - alpha)
    c = (w * l + r * k) * (1 - tau) - delta * k
    eq1 = beta * ((1 - tau)*(r - delta) + 1) - 1
    eq2 = w * c ** (-gamma) - a * (1 - l)**(-eps)
    return np.array([eq1, eq2])

#Problem 8
def euler_error(knext, *args):
    old_pol, params, k_exgrid, z_exgrid, eps_grid = args
    alpha, beta, rho, sigma = params
    prob = 1/(eps_grid.shape[0])
#    k_nnext = old_pol(knext)
#    k_exgrid, z_exgrid = np.meshgrid(kgrid, zgrid)
#    k_exgrid, z_exgrid = k_exgrid.flatten(), z_exgrid.flatten()
#    error_mat = np.array((zgrid.shape[0], kgrid.shape[0]))
#    print(k_exgrid.shape, knext.shape)
    lhs = 1/(np.e**z_exgrid * k_exgrid**alpha - knext)
    rhs = 0
    for eps in eps_grid:
        z_next = rho * z_exgrid + eps * sigma
        print(z_next)
#        print(z_next)
        k_nnext = old_pol(knext, z_next)
        rhs += beta * prob * ((alpha * np.e ** z_next * knext **(alpha - 1))/\
                              (np.e**z_next * knext**alpha - k_nnext))
    return ((lhs - rhs)**2).sum()

def find_opt_val(params):
    alpha, beta, rho, sigma = params
    eps_grid = np.array([(3/2)**0.5, 0, -(3/2)**0.5])
    k_bar = (alpha * beta)**(1/(1-alpha))
    kup, klow = 1.5 * k_bar, 0.5*k_bar
    kgrid = np.linspace(kup, klow, 26)
    zgrid = np.linspace(-5 * sigma, 5*sigma, 26)
    k_exgrid, z_exgrid = np.meshgrid(kgrid, zgrid)
    k_now = k_exgrid.copy().flatten()
    k_next = np.zeros(k_exgrid.shape) + klow
    k_next = k_next.flatten()
    old_pol = interp.LinearNDInterpolator((k_exgrid.flatten(), z_exgrid.flatten())\
                                          , k_next)
    V = np.zeros(kgrid.shape[0] * zgrid.shape[0])
    while scipy.linalg.norm(k_now - k_next) > 1e-5:
#        print(k_next.shape)
        print('success')
        param_args = (old_pol, params, k_exgrid.flatten(), z_exgrid.flatten(), eps_grid)
        k_now = k_next.copy()
        k_next = opt.minimize(euler_error, x0 = k_next, args = (param_args), \
                            method = 'L-BFGS-B', tol = 1e-15).x
        k_next = k_next
        print(k_next)
        old_pol = interp.LinearNDInterpolator((k_exgrid.flatten(), z_exgrid.flatten())\
                                          , k_next)
        V = np.log(np.e ** z_exgrid.flatten() * k_exgrid.flatten() ** alpha - k_next) + beta * V
    return k_next.reshape(26,26), k_exgrid, V.reshape(26, 26), old_pol
        
    


if __name__ == '__main__':
    prob5_args = np.array([2.5, 0.98, 0.4, 0.1, 0, 0.05])
    result = opt.fsolve(eq5, x0 = 1, args = prob5_args, xtol = 1e-10)
    
    prob6_args = np.array([2.5, 0.98, 0.4, 0.1, 0, 0.05, 0.5, 1.5])
    result6 = opt.fsolve(eq6, x0 = np.array([5, 0.5]), args = prob6_args, xtol = 1e-10)
    
    params = np.array([0.35, 0.98, 0.95, 0.02])
    k_next, k_exgrid, V, old_pol = find_opt_val(params)
    