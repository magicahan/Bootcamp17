#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 12:42:38 2017

@author: luxihan
"""
import numpy as np
import scipy

def get_adj_mat(file_name, N):
    A = np.zeros((N, N))
    with open(file_name) as f:
        i = 0
        for line in f:
            if i == 0:
                i += 1
                continue
            i += 1
            line_list = line.split()
            from_node = int(line_list[0])
            to_node = int(line_list[1])
            A[from_node, to_node] = 1
    return A

def process_matrix(A):
    Am = A.copy()
    for i in range(A.shape[0]):
        if Am[i, :].sum() == 0:
            Am[i, :] = 1
    D = Am.sum(axis = 1)
    K = (1/D).reshape((A.shape[0], 1)) * Am
    K = K.T
    return Am, D, K

def compute_it_pr(A, tol, N = None, d = 0.85):
    if N is not None:
        A_comp = A[:N, :N]
    else:
        A_comp = A.copy()
    Am, D, K = process_matrix(A_comp)
    p_init = np.random.rand(Am.shape[0], 1)
    print(p_init)
    p_init /= p_init.sum()
    p_now = p_init
    error = tol + 10
    while error > tol:
        p_prev = p_now
        p_now = d * K @ p_prev + (1 - d)/p_now.shape[0] * np.ones((Am.shape[0], 1))
        error = np.linalg.norm (p_now - p_prev)
    return p_now
    
def compute_eig_pr(A, tol, N = None, d = 0.85):
    if N is not None:
        A_comp = A[:N, :N]
    else:
        A_comp = A.copy()
    Am, D, K = process_matrix(A_comp)
    B = d * K + (1 - d)/Am.shape[0] * np.ones((Am.shape[0], Am.shape[0]))
    eigs, vecs = scipy.linalg.eig(B)
#    for i in range(eigs.shape[0]):
#        if eigs[i] == 1:
#            p = vec[:, i]
    p = vecs[:, 0] / (vecs[:, 0].sum())
    return p

def process_data(file_name, tol, d = 0.7):
    team_set = set()
    result_list = []
    with open(file_name) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            else:
                team1, team2 = line.split(',')
                team1, team2 = team1.strip(), team2.strip()
                result_list.append((team1, team2))
                team_set.update([team1])
                team_set.update([team2])
    #create dict for each team
    team_list = sorted(list(team_set))
    team_dict = {}
    order_dict = {}
    A = np.zeros((len(team_list), len(team_list)))
    for i, team in enumerate(team_list):
        team_dict[team] = i
        order_dict[i] = team
    for team1, team2 in result_list:
        i = team_dict[team2]
        j = team_dict[team1]
        A[i, j] = 1
    p_ss = compute_it_pr(A, tol, d = 0.85)
    
    order = np.argsort(p_ss.reshape((p_ss.shape[0],)))
    print(order)
    best_teams = [order_dict[i] for i in order]
    
    return p_ss, best_teams

if __name__ == '__main__':
    p_ss, best_teams = process_data('ncaa2013.csv', 1e-6)
    
    
    
    
    
        