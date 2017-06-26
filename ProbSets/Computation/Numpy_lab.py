#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 08:51:56 2017

@author: luxihan
"""

import numpy as np         
#Problem 4
def clear_neg_entry(arr):
    copy_array = np.copy(arr)
    copy_array[copy_array < 0] = 0
    rv = copy_array.astype(np.int64)
    return rv

#Problem 6
def find_sto_row(arr):
    rv = np.all(arr.sum(axis = 1) == 1)
    return rv
    
#Problem 7
def get_four_largest_product(grid):
    # calculate up and down
    grid1 = grid[: -3, :]
    grid2 = grid[1 : -2, :]
    grid3 = grid[2 : -1, :]
    grid4 = grid[3 : , :]
    grid_result = grid1 * grid2 * grid3 * grid4
    up_result = grid_result.max()
    
    # calculate right and left
    grid1 = grid[ :, : -3]
    grid2 = grid[ :, 1 : -2]
    grid3 = grid[ : , 2 : -1]
    grid4 = grid[ : , 3 : ]
    grid_result = grid1 * grid2 * grid3 * grid4
    right_result = grid_result.max()
    
    # calculate upper diagonal result
    grid1 = grid[ : -3, : -3]
    grid2 = grid[1 : -2, 1 : -2]
    grid3 = grid[2 : -1, 2 : -1]
    grid4 = grid[3 : , 3 : ]
    grid_result = grid1 * grid2 * grid3 * grid4
    upper_diag_result = grid_result.max()

    
    # calculate lower diagonal result
    grid1 = grid[3 : , : -3]
    grid2 = grid[2 : -1, 1 : -2]
    grid3 = grid[1 : -2, 2 : -1]
    grid4 = grid[ : -3,  3 : ]
    grid_result = grid1 * grid2 * grid3 * grid4
    lower_diag_result = grid_result.max()
    
    rv = max(up_result, right_result, upper_diag_result, lower_diag_result)
    return rv
    
if __name__ == '__main__':  
    #Problem 1
    A = np.array([[3 , -1, 4], [1, 5, -9]])
    B = np.array([[2, 6, -5, 3], [5, -8, 9, 7], [9, -2, -3, -2]])
    print('Problem 1:')
    
    AB = np.dot(A, B)
    print('AB = ')
    print(AB)
    
    #Problem 2
    A = np.array([[3, 1, 4], [1, 5, 9], [-5, 3, 1]])
    ans = - A@A@A + 9 * A@A - 15 * A
    print()
    print('Problem 2:')
    print(ans)
    
    #Problem 3
    A = np.triu(np.ones((7, 7)))
    B = - np.tril(np.ones((7, 7 ))) + 5 * np.triu(np.ones((7, 7 ))) \
                 + np.diag([1] * 7)
    print()
    print('A  = ')
    print(A)
    print('B = ')
    print(B)
    
    #Problem 5
    A = np.arange(6).reshape((3, 2)).T
    B = np.tril(np.ones((3, 3)) * 3)
    C = np.eye(3) * -2
    first_zero = np.zeros((A.shape[1], A.shape[1]))
    second_zero = np.zeros((A.shape[0], A.shape[0]))
    third_zero = np.zeros((A.shape[0], C.shape[1]))
    fourth_zero = np.zeros((B.shape[0], A.shape[0]))
    diag_mat = np.eye(A.shape[1])
    first_row = np.hstack((A.T, first_zero, diag_mat))
    second_row = np.hstack((A, second_zero, third_zero))
    third_row = np.hstack((B, fourth_zero, C))
    rv = np.vstack((first_row, second_row, third_row))
    print()
    print('Problem 5:')
    print(rv)
    
    #Problem 7
    grid = np.load('grid.npy')
    largest_product = get_four_largest_product(grid)  
    print()
    print('Problem 7')
    print('The largest adjacent number product is {}'.format(largest_product))
    
    
    
    
    