#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 11:26:58 2017

@author: luxihan
"""
import calculator as calc
import random

def output_list(li):
    rv = [min(li), max(li), sum(li) / len(li)]
    return rv
    
#Problem 3
def compute_hypotenuse(a, b):
    c = calc.Sqrt(calc.Sum(calc.Product(a, a), calc.Product(b, b)))
    return c

#Problem 4
def shutbox():
    status = True
    num_list = list(range(1, 10))
    name = input('Your name:')
    while status:
        if len(num_list) == 0:
            print('Score for {}:\t{}points.'.format(name, 0))
            print('Congradulation! You shut the box!')
            break
        print('Numbers Left:\t\t{}'.format(num_list))
        if sum(num_list) > 6:
            dice1 =  random.randint(1, 6)
            dice2 = random.randint(1, 6)
            print('Roll:\t\t{}  {}'.format(dice1, dice2))
            numbers = input('Numbers to eliminate:\t\t').split()
            numbers = [int(i) for i in numbers]
            if sum(numbers) == (dice1 + dice2):
                for num in numbers:
                    if num in num_list:
                        num_list.remove(num)
                    else:
                        print('The game is over!')
                        print('Name:{} \t Score:{}'.format(name, sum(num_list)))
                        status = False
                        break
            else:
                print('The game is over!')
                print('Name:{} \t Score:{}'.format(name, sum(num_list)))
                status = False
        else:
            dice1 =  random.randint(1, 6)
            print('Roll:\t\t{}'.format(dice1))
            numbers = input('Numbers to eliminate:\t\t').split()
            numbers = [int(i) for i in numbers]
            if sum(numbers) == (dice1):
                for num in numbers:
                    if num in num_list:
                        num_list.remove(num)
                    else:
                        print('The game is over!')
                        print('Name:{} \t Score:{}'.format(name, sum(num_list)))
                        status = False
                        break
                print()
            else:
                print('The game is over!')
                print('Name:{} \t Score:{}'.format(name, sum(num_list)))
                status = False
                
if __name__ == '__main__':
    shutbox()
    #Problem 2
    
    #number
    num1 = 2
    num2 = num1
    num2 += 1
    if num1 == num2:
        print('Number is mutable')
    else:
        print('Number is immutable')
        
    #strings
    str1 = 'hello'
    str2 = str1
    str2 += 'a'
    if str1 == str2:
        print('String is mutable')
    else:
        print('String is immutable')
        
    
    #lists
    li1 = [1, 2, 3]
    li2 = li1
    li2.append(1)
    if li1 == li2:
        print('List is mutable')
    else:
        print('List is immutable')
        
    #tuple
    tup1 = (1, 2)
    tup2 = tup1
    tup2 += (1, )
    if tup1 == tup2:
        print('Tuple is mutable')
    else:
        print('Tuple is immutable')
        
    #dictionary
    dic1 = {2: 'b', 3:'c'}
    dic2 = dic1
    dic2[1] = 'a'
    if dic1 == dic2:
        print('Dictionary is mutable')
    else:
        print('Dictionary is immutable')
            
    
    
    
    
    
    
    