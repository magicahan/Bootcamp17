#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 00:06:06 2017

@author: luxihan
"""
import numpy as np
import scipy
from functools import reduce

#Problem 1
def modified_qr(A):
    m, n = A.shape
    A = A.astype(float)
    Q = A.copy()
    R = np.zeros((n, n))
    for i in range(n):
        R[i, i] = scipy.linalg.norm(Q[:, i]) 
        Q[:, i] = (Q[:, i] / R[i, i])
        print(Q)
        for j in range(n - i - 1):
            R[i, j + i + 1] = Q[:, j + i + 1].T @ Q[:, i]
            Q[:, j + i + 1] = Q[:, j + i + 1] - R[i, j + i + 1] * Q[:, i]
    return Q, R

def compute_det(A):
    Q, R = modified_qr(A)
    diag_element = np.diag(R)
    product = reduce((lambda x, y: x * y), diag_element)
    return product

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
            
def householder(A):
    A = A.astype(float)
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)
    for k in range(n):
        u = (R[k:, k].copy()).reshape(n - k, 1)
        u[0, 0] += np.sign(u[0, 0]) * scipy.linalg.norm(u)
        u /= scipy.linalg.norm(u)
        R[k:, k:] -= 2 * u @ (u.T @ R[k:, k:])
        Q[k:, :] -= 2 * u @ (u.T @ Q[k:, :])
    return Q.T, R                    
            
def hessenberg(A):
    A = A.astype(float)
    m, n = A.shape
    H = A.copy()
    Q = np.eye(m)
    for k in range(n - 2):
        u = H[k + 1:, k].copy()
        u = u.reshape(len(u), 1)
        u[0, 0] += np.sign(u[0, 0]) * scipy.linalg.norm(u)
        u /= scipy.linalg.norm(u)
        H[k + 1:, k:] -= 2 * u @ (u.T @ H[k+1:, k:])
        H[:, k + 1:] -= 2 * (H[:, k + 1:] @ u) @ u.T
        Q[k + 1:, :] -= 2 * u @ (u.T @Q[k + 1:, :])
    return H, Q.T
    
            
            
            