#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 22:05:15 2017

@author: luxihan
"""
import numpy as np
import scipy
import sympy as sy

import matplotlib.pyplot as plt

def compute_condnum(A):
    sing_vals = scipy.linalg.svdvals(A)
    sig_max = max(sing_vals)
    sig_min = min(sing_vals)
    if sig_min == 0:
        return np.inf
    else:
        return sig_max / sig_min
    
    
def perturb_w():
    w_roots = np.arange(1, 21)
    x, i = sy.symbols('x i')
    w = sy.poly_from_expr(sy.product(x-i, (i, 1, 20)))[0]
    w_coeffs = np.array(w.all_coeffs())
    cond_list = []
    print(w_coeffs)
    
    plt.scatter(w_roots, np.zeros(w_roots.shape[0]), s = 20)
    
    for i in range(100):
        h = np.random.normal(loc = 1, scale = 1e-10, size = w_coeffs.shape[0])
        new_coeffs = w_coeffs*h
        new_roots = np.roots(np.poly1d(new_coeffs))
        reals = np.real(new_roots)
        imags = np.imag(new_roots)
        plt.scatter(reals, imags, marker = ',', s = 1)
        k_new = scipy.linalg.norm(new_roots - w_roots, np.inf)/\
                            scipy.linalg.norm(h, np.inf)
        cond_list.append(k_new)
    plt.show()
    return cond_list

def compute_cond_eig(A):
    reals = np.random.normal(0, 1e-10, A.shape)
    imags = np.random.normal(0, 1e-10, A.shape)
    H = reals + 1j*imags
    eigs = scipy.linalg.eigvals(A)
    eigs_new = scipy.linalg.eigvals(A + H)
    cond = np.linalg.norm(eigs - eigs_new, 2) / np.linalg.norm(H, 2)
    rela_cond = cond * np.linalg.norm(A, 2)/np.linalg.norm(eigs, 2)
    return cond, rela_cond

#Problem 4
def plot_mesh_eig(bounds, res):
    x_min, x_max, y_min, y_max = bounds
    x = np.linspace(x_min, x_max, res)
    y = np.linspace(y_min, y_max, res)
    X, Y = np.meshgrid(x, y)
#    X, Y = X.flatten(), Y.flatten()
    cond = np.zeros((X.shape[0], Y.shape[1]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            xi = X[i, j]
            yi = Y[i, j]
            A = np.ones((2, 2))
            A[0, 1] = xi
            A[1, 0] = yi
            cond_num, rela_cond_num = compute_cond_eig(A)
            cond[i, j] = rela_cond_num
    print(X.shape, cond.shape)
    plt.pcolormesh(X, Y, cond, cmap = 'gray_r')
    plt.show()
    return X, Y ,cond

#Problem 5
def plot_poly(n):
    xk, yk = np.load('stability_data.npy').T
    A = np.vander(xk, n + 1)
    x_range = np.linspace(np.min(xk), np.max(xk), 500)
    
    #solve for the inverse method
    x_inv = scipy.linalg.inv(A.T@A)@A.T@yk
    y_inv_range = np.polyval(x_inv, x_range)
    error_inv = scipy.linalg.norm(A @ x_inv - yk, 2)
    
    #solve for the QR method
    q, r = scipy.linalg.qr(A, mode = 'economic')
    x_qr = scipy.linalg.solve_triangular(r, q.T@yk)
    y_qr_range = np.polyval(x_qr, x_range)
    error_qr = scipy.linalg.norm(A @ x_qr - yk, 2)
    
    plt.scatter(xk, yk)
    plt.plot(x_range, y_inv_range, label = 'Inverse Approxmation', lw = 2)
    plt.plot(x_range, y_qr_range, label = 'QR Approxmation', lw = 2)
    plt.ylim((0, 30))
    plt.legend(loc = 'upper left')
    plt.show()
    plt.close()
    
    return error_inv, error_qr
    
def compare_integral():
    #for sympy integrate
    x = sy.Symbol('x')
    n_list = list(range(5, 51, 5))
    int_or = []
    int_approx = []
    error_list = []
    for n in n_list:
        x1 = sy.integrate(x**n * np.e**(x - 1), (x, 0, 1))
        x2 = (-1)**n * sy.subfactorial(n) + (-1)**(n+1) * sy.factorial(n)/np.e
        int_or.append(x1)
        int_approx.append(x2)
        error = abs(x1 - x2)/abs(x1)
        error_list.append(error)
    plt.plot(n_list, error_list)
    plt.xlabel('N')
    plt.ylabel('Relative Forward Error')
    return error_list

if __name__ == '__main__':
    bounds = np.array([-5, 5, -5, 5])
    plot_mesh_eig(bounds, 200)
    
    plot_poly(14)
    
    compare_integral()
    