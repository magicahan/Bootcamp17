#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 17:54:16 2017

@author: luxihan
"""
import numpy as np
import scipy

def judge_drazin(A, k, AD):
    state = True
    if not np.allclose(A@AD, AD@A):
        state =False
    if not np.allclose(np.linalg.matrix_power(A, k + 1)@AD, \
                       np.linalg.matrix_power(A, k)):
        state = False
    if not np.allclose(AD@A@AD, AD):
        state = False
    return state

def drazin(A, tol):
    f1 = lambda x: abs(x) > tol
    f2 = lambda x: abs(x) <= tol
    n = A.shape[0]
    Q1, S, k1 = scipy.linalg.schur(A, sort=f1)
    Q2, T, k2 = scipy.linalg.schur(A, sort=f2)
    U = np.column_stack((S[:, :k1], T[:, : n - k1]))
    U_inv = np.linalg.inv(U)
    V = U_inv@A@U
    Z = np.zeros((n, n))
    if k1 != 0:
        M_inv = np.linalg.inv(V[:k1, :k1])
        Z[:k1, :k1] = M_inv
    return U@Z@U_inv

def effective_distance(A, tol):
    n = A.shape[0]
    E = np.eye(n)
    R = np.zeros((n, n))
    L = scipy.sparse.csgraph.laplacian(A)
    for j in range(n):
        L_j = L.copy()
        L_j[j, :] = E[j, :]
        L_drazin = drazin(L_j, tol)
        for i in range(n):
            if i != j:
                R[i, j] = L_drazin[i, i]
            else:
                R[i, j] = 0
    return R

class LinkPredictor():
    def __init__(self, file_name):
        self.file = file_name
        self.matrix = None
        self.name_list = None
        self.effective_resistance = None
        self.graph_list = None
        self.name_dict = None
    
    def process(self):
        graph_list = []
        name_list = []
        name_dict = {}
        with open(self.file) as f:
            for line in f:
                name1, name2 = line.split(',')
                name1, name2 = name1.strip(), name2.strip()
                graph_list.append((name1, name2))
                if name1 not in name_list:
                    name_list.append(name1)
                if name2 not in name_list:
                    name_list.append(name2)
        name_list = sorted(name_list)
        
        k = 0
        for name in name_list:
            name_dict[name] = k
            k += 1
            
        self.graph_list = graph_list
        self.name_dict = name_dict
        self.name_list = name_list
        
    def get_matrix(self):
        self.process()
        n = len(self.name_dict.keys())
        ad_matrix = np.zeros((n, n))
        for name_tup in self.graph_list:
            name1, name2 = name_tup
            i, j = self.name_dict[name1], self.name_dict[name2]
            ad_matrix[i, j] += 1
            ad_matrix[j, i] += 1
        self.matrix = ad_matrix
        
    def compute_effective_resis(self, tol):
        self.effective_resistance = effective_distance(self.matrix, tol)
        
    def predict_link(self, node):
        if node == None:
            eff_copy = self.effective_resistance.copy()
            eff_copy[eff_copy == 0] = eff_copy.max() + 10
            minval = np.min(eff_copy)
            loc = np.where(eff_copy == minval)
            print(loc)
            loc = (int(loc[0]), int(loc[1]))
        
        else:
            row_index = self.name_dict[node]
            n = self.matrix.shape[0]
            eff_copy = np.zeros((n, n)) + np.max(self.effective_resistance) + 10
            eff_copy[row_index, :] = self.effective_resistance[row_index, :]
            eff_copy[eff_copy == 0] = np.max(self.effective_resistance)
            minval = np.min(eff_copy)
            loc = np.where(eff_copy == minval)
            loc = (int(loc[0]), int(loc[1]))
        for name in self.name_dict.keys():
            if self.name_dict[name] == loc[0]:
                name1 = name
            elif self.name_dict[name] == loc[1]:
                name2 = name
        return (name1, name2)
        
    def add_link(self, name_tup):
        name1, name2 = name_tup
        if name1 not in self.name_list or name2 not in self.name_list:
            raise ValueError('Name not in network')
        index1 = self.name_dict[name1]
        index2 = self.name_dict[name2]
        
        self.matrix[index1, index2] = 1
        self.matrix[index2, index1] = 1
        
        self.compute_effective_resis(1e-6)

if __name__ == '__main__':
    tol = 1e-8
    A = np.array([[1, 3, 0, 0], [0, 1, 3, 0], [0, 0, 1, 3], [0, 0,0 ,0]])
    AD = drazin(A, tol)
    
    graph = LinkPredictor('social_network.csv')
    graph.process()
    graph.get_matrix()
    graph.compute_effective_resis(tol)
    
    
    
    