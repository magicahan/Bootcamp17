#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:58:33 2017

@author: luxihan
"""
import numpy as np
import scipy
from functools import reduce
import matplotlib.pyplot as plt

def truncated_svd(A, k = None):
    AH = (np.matrix(A)).getH()
    eigs, vecs = scipy.linalg.eig(AH@A)
    if k == None:
        sort_index = np.argsort(eigs)[::-1]
        eig_sort = eigs[sort_index]
        r = sum(eig_sort != 0)
        sigma = np.diag(eig_sort[:r])**0.5
        sig_inv = np.diag(1 / eig_sort[:r])**0.5
        V = vecs[:, sort_index[:r]]
        U = A @ V @ sig_inv
        return U, sigma, V
    else:
        sort_index = np.argsort(eigs)[::-1]
        eig_sort = eigs[sort_index]
        sigma = np.diag(eig_sort[:k])**0.5
        sig_inv = np.diag(1 / eig_sort[:r])**0.5
        V = vecs[:, sort_index[:k]]
        U = A @ V @ sig_inv
        return U, sigma, V
    

def plot_trans(A, k):
    #decompose A
    u, sigma, v = truncated_svd(A, k)
    domain_space = np.linspace(0, 2 * np.pi, 1000)
    x = np.cos(domain_space)
    y = np.sin(domain_space)
    
    base = np.array([[1, 0], [0, 1]])
    
    xy = np.vstack((x, y))
    
    #original plot
    plt.subplot(2, 2, 1)
    plt.plot([0, base[0,0]], [0, base[1, 0]], color = 'red')
    plt.plot([0, base[0,1]], [0, base[1, 1]], color = 'red')
    plt.plot(xy[0, :], xy[1, :], color = 'blue')
    plt.axis('equal')
    
    #First transformation
    v = np.matrix(v)
    print(xy)
    xy = np.array(v.getH() @ xy)
    print(xy)
    base = v.getH()@base
    
    plt.subplot(2, 2, 2)
    plt.plot([0, base[0,0]], [0, base[1, 0]], color = 'red')
    plt.plot([0, base[0,1]], [0, base[1, 1]], color = 'red')
    plt.plot(xy[0, :], xy[1, :], color = 'blue')
    plt.axis('equal')
    
    #Second Transformation
    xy = np.array(sigma @ xy)
    base = sigma@base
    
    plt.subplot(2, 2, 3)
    plt.plot([0, base[0,0]], [0, base[1, 0]], color = 'red')
    plt.plot([0, base[0,1]], [0, base[1, 1]], color = 'red')
    plt.plot(xy[0, :], xy[1, :], color = 'blue')
    plt.axis('equal')
    
    #Second Transformation
    xy = np.array(u @ xy)
    base = u@base
    
    plt.subplot(2, 2, 4)
    plt.plot([0, base[0,0]], [0, base[1, 0]], color = 'red')
    plt.plot([0, base[0,1]], [0, base[1, 1]], color = 'red')
    plt.plot(xy[0, :], xy[1, :], color = 'blue')
    plt.axis('equal')
    
    plt.show()
    plt.close()
    
    return xy
    
    
def svd_approx(A, k):
    U,s,Vh = scipy.linalg.svd(A, full_matrices=False)
    S = np.diag(s[:k])
    Ahat = U[:,:k].dot(S).dot(Vh[:k,:])
    return Ahat

def lowest_rank_approx(A, e):
    U,s,Vh = scipy.linalg.svd(A, full_matrices=False)
    k = 0
    for i in range(s.shape[0]):
        if s[i, i] >= e:
            k += 1
        else:
            break
    Ahat = svd_approx(A, k)
    return Ahat

def compress_image(file_name, k):
    R = plt.imread(file_name)[:,:,0].astype(float)
    G = plt.imread(file_name)[:,:,1].astype(float)
    B = plt.imread(file_name)[:,:,2].astype(float)
    
    R = R/255
    G = G/255
    B = B/255
    plt.subplot(1, 2, 1)
    plt.imshow(np.dstack((R, G, B)))
    
    R = svd_approx(R, k)
    G = svd_approx(G, k)
    B = svd_approx(B, k)
    
    X = np.dstack((R, G, B))
    plt.subplot(1, 2, 2)
    plt.imshow(X)
    
if __name__=='__main__':
    #Problem 2
    A = np.array([[3, 1], [1, 3]])
    plot_trans(A, None)
    
    
    #Problem 5
    compress_image('hubble.jpg', 20)

