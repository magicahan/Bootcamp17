#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 10:49:57 2017

@author: luxihan
"""
from random import choice
import numpy as np
import re


def arithmagic():
    step_1 = input("Enter a 3-digit number where the first and last "
    "digits differ by 2 or more: ")
    if len(step_1) != 3:
        raise ValueError('The number entered is not a three-digit number.')
    if abs(int(step_1[0]) - int(step_1[-1])) < 2:
        raise ValueError('The first and the last digit differ by less than 2.')
    step_2 = input("Enter the reverse of the first number, obtained by reading it backwards: ")
    if step_2 != step_1[::-1]:
        raise ValueError('The number entered is not the reverse of step 1.')
    step_3 = input("Enter the positive difference of these numbers: ")
    if step_3 != abs(int(step_1) - int(step_2)):
        raise ValueError('The number entered is not the positive difference.')
    step_4 = input("Enter the reverse of the previous result: ")
    if step_4 != step_3[::-1]:
        raise ValueError('The number entred is not the reverse of step 3.')
    print (str(step_3) + " + " + str(step_4) + " = 1089 (ta-da!)")
    
def random_walk(max_iters=1e12):
    try:
        walk = 0
        direction = [1, -1]
        for i in range(int(max_iters)):
            walk += choice(direction)
    except KeyboardInterrupt:
        print('Process interrupted at iteration i')
    else:
        print('Process completed')
    finally:
        return walk
    
class ContentFilter():
    def __init__(self, file_name):
        if type(file_name) != str:
            raise TypeError('File name is not a string.')
        self.file_name = file_name
        with open(file_name, 'r') as f:
            self.contents = f.read()
            
    def uniform(self, file_name, case, mode = 'w'):
        if mode not in ['w', 'a']:
            raise ValueError('Mode is wrong.')
        if case == 'upper':
            with open(file_name, mode) as f:
                f.write(self.contents.upper())
        elif case == 'lower':
            with open(file_name, mode) as f:
                f.write(self.contents.lower())
        else:
            raise ValueError('Case is specified wrongfully.')
        
    def reverse(self, file_name, unit = 'line', mode = 'w'):
        if mode not in ['w', 'a']:
            raise ValueError('Mode is wrong.')
        if unit == 'line':
            content_list = self.contents.split('\n')
            content_list = content_list[::-1]
            with open(file_name, mode) as f:
                f.writelines(content_list)
        elif unit == 'word':
            content_list = self.contents.split('\n')
            content_list = [line[::-1] for line in content_list]
            with open(file_name, mode) as f:
                f.writelines(content_list)
        else:
            raise ValueError('Wrong unit specified.')
            
    def transpose(self, file_name, mode = 'w'):
        if mode not in ['w', 'a']:
            raise ValueError('Mode is wrong.')
        content_list = self.contents.split('\n')
        content_list = [line.split() for line in content_list]
        content_array = np.array(content_list).T
        content_list = content_array.tolist()
        content_list = [' '.join(line_list) for line_list in content_list]
        with open(file_name, mode) as f:
            f.writelines(content_list)
            
    def __str__(self):
        num_letter = len(re.findall('[a-zA-z]', self.contents))
        num_digit = len(re.findall('\d', self.contents))
        num_white = len(re.findall('\w', self.contents))
        doc_string = 'Source file:\t{}\nTotal characters:\t{}\nAlphabetic '\
                    'characters:\t{}\nNumerical characters:\t{}\n'\
                    'Withesapce characters:\t{}\n'\
                    'Number of lines:\t{}'.format(self.file_name, \
                                       len(self.contents),
                                          num_letter, num_digit, num_white,
                                          len(self.contents.split('\n')))
        return doc_string
        
        
        
            
            
            
        

