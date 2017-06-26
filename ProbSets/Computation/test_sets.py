#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 23:55:12 2017

@author: luxihan
"""
import solutions as soln

def test_read_sets_valid(file_name):
    assert soln.read_sets(file_name), 'Invalid Input'
    
def test_read_sets_duplicate(file_name):
    assert soln.read_sets(file_name), 'Duplicate Cards'
    
def test_read_sets_cardnum(file_name):
    assert soln.read_sets(file_name), 'Insufficient amount of cards'
    
def test_read_sets_filename(file_name):
    assert soln.read_sets(file_name), 'Wrong file name'
    
if __name__ == '__main__':
    filenames = ['test_invalid.txt', 'test_duplicate.txt', 'test_filename.tex', \
                'test_num.txt']
    try:
        test_read_sets_valid(filenames[0])
    except Exception as e:
        print(e)
        pass
    try:
        test_read_sets_duplicate(filenames[1])
    except Exception as e:
        print(e)
        pass
    try:
        test_read_sets_cardnum(filenames[3])
    except Exception as e:
        print(e)
        pass
    try:
        test_read_sets_filename(filenames[2])
    except Exception as e:
        print(e)
        pass
    print(soln.compute_set('test.txt'))
        
    