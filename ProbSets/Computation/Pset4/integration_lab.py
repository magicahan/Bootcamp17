#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 22:14:34 2017

@author: luxihan
"""
import numpy as np
import scipy.stats as sts
import scipy
import scipy.optimize as opt

def newton_int(g, bound, N, method):
    a, b = limit
    if method == 'midpoint':
        result = 0
        grid = np.linspace(a, b, N + 1)
        for i in range(N):
            result += g((grid[i] + grid[i + 1]) / 2)
        return result * (b - a)/N
    elif method == 'trapezoid':
        result = 0
        grid = np.linspace(a, b, N + 1)
        for i in range(N + 1):
            result += 2 * g(grid[i])
        result -= (g(grid[0]) + g(grid[-1]))
        return (b - a)/(2*N) * result
    elif method == 'Simpsons':
        result = 0
        grid = np.linspace(a, b, 2 * N + 1)
        for i in range(2 * N + 1):
            if i == 0:
                result += g(grid[i])
            elif i == 2 * N:
                result += g(grid[i])
            elif i % 2 == 1:
                result += 4 * g(grid[i])
            elif i % 2 == 0:
                result += 2 * g(grid[i])
        return (b - a)/(3 * (N + 1)) * result
        
def approximate_dist(mu, sigma, k, N):
    def norm_cdf(x):
        return sts.norm.cdf(x, loc = mu, scale = sigma)
    def norm_pdf(x):
        return sts.norm.pdf(x, loc = mu, scale = sigma)
    grid = np.linspace(mu - k * sigma, mu + k * sigma, N)
    zw_list = []
    for i in range(N):
        if i == 0:
            z = grid[i]
            z2 = grid[i + 1]
            w = norm_cdf((z + z1)/2)
            zw_list.append((z, w))
        elif i > 0 and i < N - 1:
            z = grid[i]
            z_min = (grid[i] + grid[i - 1])/2
            z_max = (grid[i] + grid[i + 1])/2
            w = newton_int(norm_pdf, (z_min, z_max), 1000, 'midpoint')
            zw_list.append((z, w))
        elif i == N - 1:
            z = grid[i]
            w = 1 - orm_cdf((grid[i] + grid[i - 1])/2)
            zw_list.append((z, w))
    return zw_list
    

def discretize_LN(mu, sigma, k, N):
    zw_list = approximate_dist(mu, sigma, k, N)
    zw_LN_list = []
    for z, w in zw_list:
        zw_LN_list.append((np.exp(z), w))
    return zw_LN_list

def compute_us_income(k, N, mu = 10.5, sigma = 0.8):
    result = 0
    zw_LN_list = discretize_LN(mu, sigma, k, N)
    for z, w in zw_LN_list:
        result += z * w
    return result

def gaussian_q(limit):
#    q0, q1, q2, q3, q4 = params
    a, b = limit
    def find_error(vars):
        w1, w2, w3, x1, x2, x3 = vars
        w = np.array([w1, w2, w3])
        x = np.array([x1, x2, x3])
        poly0 = w.sum()
        poly1 = w @ x - (b - a)
        poly2 = w @ x**2 - 0.5 * (b ** 2 - a ** 2)
        poly3 = w @ x**3 - 1/3 * (b ** 3 - a ** 3)
        poly4 = w @ x**4 - 1/4 * (b**4 - a**4)
        poly5 = w @ x**5 - 1/5 * (b**5 - a**5)
        return np.array([poly0, poly1, poly2, poly3, poly4, poly5])
    result = opt.fsolve(find_error, x0 = np.ones(6) * 0.1, \
#                        method = 'L-BFGS-B',
                        xtol = 1e-6)
    return result

if __name__ == '__main__':
    result = gaussian_q((-10, 10))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    