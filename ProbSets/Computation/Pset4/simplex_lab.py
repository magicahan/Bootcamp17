#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 20:11:11 2017

@author: luxihan
"""

import numpy as np

class SimplexSolver():
    def __init__(self, A, c, b):
        self.A = A
        self.c = c
        self.b = b
        if (b <= 0).sum() >= 1:
            raise ValueError('Origin is not feasible.')
        self.index_list = []
        self.var_list = []
            
    def gen_index_list(self):
        m = self.c.shape[0]
        n = self.A.shape[0]
        slack = list(range(n, n + m))
        primal = list(range(n))
        self.index_list = slack + primal
    
    
    def adjust_index(self, m):
        basic_list = []
        non_basic_list = []
        for i in range(len(self.var_list)):
            if self.var_list[i] != 0:
                non_basic_list.append(i)
            else:
                basic_list.append(i)
        max_non_basic = max(non_basic_list)
        min_basic = min(basic_list)
        while max_non_basic > min_basic:
            self.var_list[max_non_basic], self.var_list[min_basic] = \
                    self.var_list[min_basic], self.var_list[max_non_basic]
            basic_list.remove(min_basic)
            non_basic_list.remove(max_non_basic)
            if basic_list == [] or non_basic_list == []:
                break
            else:
                max_non_basic = max(non_basic_list)
                min_basic = min(basic_list)
    
    def gen_tab(self):
        m, n= self.A.shape
        A_bar = np.column_stack((self.A, np.eye(m)))
        c_bar = np.append(-self.c, np.zeros(m))
        col1 = np.append(np.array([0]), self.b)
        col3 = np.append(np.array([1]), np.zeros(m))
        col2 = np.vstack((c_bar, A_bar))
        self.T = np.column_stack((col1, col2, col3))
        
    def swap_index(self, enter, leave):
        index1 = self.index_list.index(enter)
        index2 = self.index_list.index(leave)
        self.index_list[index1], self.index_list[index2] =\
                       self.index_list[index2], self.index_list[index1]
        
    def pivot_leaving(self):
        n_row, n_col = self.T.shape
        for i in range(n_col):
            if i == 0:
                continue
            if self.T[0, i] < 0:
                p_col = i
                break
        
        #select positive element
        pos_ele = []
        for i in range(n_row):
            if i == 0:
                continue
            if self.T[i, p_col] > 0:
                pos_ele.append(i)
        
        if len(pos_ele) == 1:
            p_row = pos_ele[0]
        else:
            ratio_list = []
            for i in pos_ele:
                ratio_list.append((self.T[i, 0] / self.T[i, p_col], i))
#            print(ratio_list)
            best_ratio = ratio_list[0][0]
            best_index = ratio_list[0][1]
            for ratio, index in ratio_list:
                if ratio < best_ratio:
                    best_index = index
                    best_ratio = ratio
                elif ratio == best_ratio:
                    best_index = min(index, best_index)
            p_row = best_index
        return p_col - 1, p_row + 1
                
    
    def row_operation(self):
        p_col, p_row = self.pivot_leaving()
        print(p_col, p_row)
        self.swap_index(p_col, p_row)
        p_col, p_row = p_col + 1, p_row - 1
        n_row, n_col = self.T.shape
        if self.T[p_row, p_col] == 0:
            raise ValueError('Unboudned Error')
        for i in range(n_row):
            if self.T[i, p_col] != 0 and i != p_row:
                self.T[i, :] -= self.T[i, p_col]/self.T[p_row, p_col] * \
                      self.T[p_row, :]
            elif i == p_row:
                self.T[p_row, :] /= self.T[p_row, p_col]
                      
    def solve(self):
        self.gen_index_list()
        self.gen_tab()
        self.opt_nbasic = {}
        self.opt_basic = {}
        state = (self.T[0,:] < 0).sum()
        while state:
            self.row_operation()
            state = (self.T[0,:] < 0).sum()
        for i in range(self.T.shape[0] - 1):
            self.opt_nbasic[self.index_list[i]] = self.T[i + 1, 0]
        for index in self.index_list[self.T.shape[0] - 1 : ]:
            self.opt_basic[index] = 0
        self.opt_sol = (self.T[0, 0], self.opt_nbasic, self.opt_basic)
        return self.opt_sol
    
    
def solve_demand(file_name):
    file_cont = np.load(file_name)
    A, c, b = file_cont['A'], file_cont['p'], file_cont['m']
    solver = SimplexSolver(A, c, b)
    cost, nbasic, basic = solver.solve()
#    print(nbasic, basic)
    unit_index_list = list(range(c.shape[0]))
    unit = np.zeros(c.shape[0])
    for key, item in nbasic.items():
        if key >= b.shape[0]:
            unit[key - b.shape[0]] = item
    for key, item in basic.items():
        if key >= b.shape[0]:
            unit[key - b.shape[0]] = item
    unit = np.minimum(unit, file_cont['d'])
    return unit
    
    
if __name__=='__main__':
    c = np.array([3., 2.])
    b = np.array([2., 5, 7])
    A = np.array([[1, -1], [3, 1], [4, 3]])
    solver = SimplexSolver(A, c, b)
    sol = solver.solve()
    unit = solve_demand('productMix.npz')
    print(unit)
                