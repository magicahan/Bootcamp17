#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 19:39:34 2017

@author: luxihan
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy

def newton(f, fp, x0, alpha = 1, tol = 1e-5, maxiter = 15):
    x1 = x0
    x2 = x0
    for i in range(maxiter):
        x1 = x2
        x2 = x1 - alpha * f(x1)/fp(x1)
        if abs(x1 - x2) <= tol:
            return x2, True
    if abs(x1 - x2) <= tol:
        return x2, True
    else:
        return x2, False
    

def compute_r(P1, P2, N1, N2):
    def balance_func(r):
        return P1 * ((1 + r)**N1 - 1) - P2 * (1 - (1 + r) ** (-N2))
    def b_fp(r):
        return P1*N1*(1 + r)**(N1 - 1) - P2 * N2 * (1 + r)**(-N2 - 1)
    r, state = newton(balance_func, b_fp, 0.1)
    return r, state

def plot_alpha(f, fp):
    def newton(f, fp, x0, alpha = 1, tol = 1e-5, maxiter = 150):
        x1 = x0
        x2 = x0
        for i in range(maxiter):
            x1 = x2
            x2 = x1 - alpha * f(x1)/fp(x1)
            if abs(x1 - x2) <= tol:
                return x2, i + 1
        if abs(x1 - x2) <= tol:
            return x2, maxiter + 1
        else:
            return x2, maxiter + 1
    alpha_arr = np.linspace(1e-10, 1, 1000)
    iter_list = []
    for alpha in alpha_arr:
        x2, iter = newton(f, fp, .01, alpha = alpha)
        iter_list.append(iter)
    plt.plot(alpha_arr, iter_list)
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Num of Iterations')
    plt.show()
    
    
def newton_higher(f, fp, x0, alpha = 1, tol = 1e-5, maxiter = 15):
    if np.issccalar(x0):
        return newton(f, fp, x0, alpha, tol, maxiter)
    else:
        x1 = x0
        x2 = x0
        for i in range(maxiter):
            x1 = x2
            Df = fp(x1)
            y = scipy.linalg.solve(Df, f(x1))
            x2 = x1 - alpha * y
            if scipy.linalg.norm(x1 - x2) <= tol:
                return x2, True
        if scipy.linalg.norm(x1 - x2) <= tol:
            return x2, True
        else:
            return x2, False
        
def f6(x, y, params):
    gamma, delta = params
    return np.array([gamma * x * y - x * (1 + y), - x * y + (delta -y)*(1 + y)])


def f6p(x, y, params):
    gamma, delta = params
    x11 = gamma * y - (1 + y)
    x12 = gamma * x - x
    x21 = -y
    x22 = -x - (1 + y) + (delta - y)
    return np.array([[x11, x12], [x21, x22]])

def find_initial(f, fp, result, xlim, ylim):
    xl, xu = xlim
    yl, yu = ylim
    xgrid = np.linspace(xl, xu, 10000)
    ygrid = np.linspace(yl, yu, 10000)
    for x in xgrid:
        for y in ygrid:
            x1, state = newton_higher(f, fp, np.array([x, y]), alpha = 0.55)
            if scipy.linalg.norm(result - x1) <= 1e-5:
                return np.array([x, y])
    return None
        


if __name__=='__main__':
    N1, N2, P1, P2 = 30, 20, 2000, 8000
    r, state = compute_r(P1, P2, N1, N2)
    
    f = lambda x: np.sign(x) * np.power(np.abs(x), 1./3)
    f_p = lambda x:  1./3 * np.power(np.abs(x), -2./3)
    
    x1, state1 = newton(f, f_p, .01)
    x2, state2 = newton(f, f_p, .01, 0.4)
    
    plot_alpha(f, f_p)
    