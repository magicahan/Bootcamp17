#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 08:16:57 2017

@author: luxihan
"""
import numpy as np
import scipy
from functools import reduce
import matplotlib.pyplot as plt
import cmath

def modified_qr(A):
    m, n = A.shape
    A = A.astype(float)
    Q = A.copy()
    R = np.zeros((n, n))
    for i in range(n):
        R[i, i] = scipy.linalg.norm(Q[:, i]) 
        Q[:, i] = (Q[:, i] / R[i, i])
        for j in range(n - i - 1):
            R[i, j + i + 1] = Q[:, j + i + 1].T @ Q[:, i]
            Q[:, j + i + 1] = Q[:, j + i + 1] - R[i, j + i + 1] * Q[:, i]
    return Q, R

def solve_linear(A, b):
    Q, R = modified_qr(A)
    y = Q.T @ b
    n = R.shape[0]
    x = np.zeros((n , 1))
    for i in range(n):
        if i == 0:
            x[n - i - 1, 0] = y[n - i - 1] / R[n - i - 1, n - i -1]
            print((n - i -1, x[n - i - 1, 0]))
        else:
            print(R[n - i - 1, :] @ x)
            x[n - i- 1, 0] = (y[n - i - 1] - R[n - i - 1, :] @ x) / \
                     R[n - i - 1, n - i - 1]
    return x

#problem 2
def compute_ols(ind, dep):
    ind = np.column_stack((ind, np.ones(ind.shape[0]).reshape(ind.shape[0], 1)) )
    coef = solve_linear(ind, dep)
    return coef

def preprocess(file_name):
    data = np.load(file_name)
    ind = data[:, 0]
    ind = ind.reshape(ind.shape[0], 1)
    dep = data[:, 1]
    print(ind)
    return ind, dep

def compute_poly(ind, dep):
    A = np.column_stack((ind, ind ** 3, ind ** 6, \
                         ind ** 9, ind ** 12))
    coef = scipy.linalg.lstsq(A, dep)[0]
    return coef

#Problem 3
def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A))/(2*A)
    plt.plot(r*cos_t, r*sin_t, lw=2)
    plt.gca().set_aspect("equal", "datalim")
    
def fit_elip(file_name):
    data = np.load(file_name)
    xy = data[:, 0] * data[:, 1]
    ind = np.column_stack((data**2, xy, data))
    dep = np.ones(data.shape[0])
    coef = scipy.linalg.lstsq(ind, dep)[0]
    a, e, c, b, d = coef
    plot_ellipse(a, b, c, d, e)
    plt.scatter(data[:, 0], data[:, 1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Fitted Ellipse')
    plt.show()
    plt.close()
    
def power(A, max_iter, tol):
    m, n = A.shape
    x_now = np.random.rand(n)
    x_now = x_now / scipy.linalg.norm(x_now)
    x_prev = x_now -1
    iter = 0
    while sum(x_now - x_prev) > tol and max_iter > iter:
        iter += 1
        x_prev = x_now
        x_now = A @ x_now
        x_now = x_now / scipy.linalg.norm(x_now)
    return x_now.T@A@x_now, x_now

def qr_algo(A, N, tol):
    m,n = A.shape
    S = scipy.linalg.hessenberg(A)
    for k in range(N):
        Q,R = modified_qr(S)
        S = R@Q
    eigs = []
    i = 0
    while i < n:
        if S[i, i + 1] == 0 and S[i + 1, i] == 0:
            eigs.append(S[i, i])
        else:
            S_i = S[i : i+2, i : i+2]
            a, b, c, d = S_i[0, 0], S_i[0, 1], S_i[1, 0], S_i[1, 1]
            eig1 = (a + d + cmath.sqrt((a+d)**2 - 4 * (a*d - b*c)))/2
            eig2 = (a + d - cmath.sqrt((a+d)**2 - 4 * (a*d - b*c)))/2
            eigs += [eig1, eig2]
            i += 1
        i += 1
    return eigs
            

if __name__ == '__main__':
    ind, dep = preprocess('housing.npy')
    coef = compute_ols(ind, dep)
    
    coef = compute_poly(ind, dep)
    plt.scatter(ind, dep)
    x = np.linspace(ind[:, 0].max(), ind[:, 0].min(), 1000)
    x = np.column_stack((x, x ** 3, x** 6, \
                         x ** 9, x ** 12))
    y = x @ coef
    plt.plot(x[:, 0], y)
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.show()
    plt.close
    
    poly_coef = compute_poly(ind, dep)
    
    fit_elip('ellipse.npy')
    
    A = np.random.random((10,10))
    lamb, v = power(A, 1000, 1e-10)
    eigs, vecs = scipy.linalg.eig(A)
    loc = np.argmax(eigs)
    lamb, x = eigs[loc], vecs[:,loc]
    print(np.allclose(A.dot(x), lamb*x))
    
    eigs2 = qr_algo(A + A.T, 1000, 1e-10)
    eigs2_test, vec_2 = scipy.linalg.eig(A + A.T)


    
    
    
    
    

