#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 21:40:14 2017

@author: luxihan
"""
import numpy as np
import scipy
import sympy as sy
import matplotlib.pyplot as plt
import autograd.numpy as anp
import autograd
import quantecon as qe

f = lambda x: (np.sin(x) + 1)**(np.sin(np.cos(x)))
x = sy.Symbol('x')
f_prime_exp = sy.diff((sy.sin(x) + 1)**(sy.sin(sy.cos(x))), x)

g = lambda x: np.log((np.sin(x**0.5))**0.5)
g_prime_exp = sy.diff(sy.log((sy.sin(x**0.5))**0.5), x)
g_prime = lambda y: sy.N(f_prime_exp.subs(x, y))

def f_prime(x_arr):
    y = np.zeros(x_arr.shape[0])
    for i, n in enumerate(x_arr):
        y[i] = sy.N(f_prime_exp.subs(x, 1))
    return y

def plot_f():
    domain = np.linspace(-np.pi, np.pi, 1000)
    f_range = f(domain)
    fp_range = f_prime(domain)
    plt.plot(domain, f_range)
    plt.plot(domain ,fp_range)
    plt.show()
    
def approx_fp(f, x, h, type, n):
    if type == 'f':
        if n == 1:
            return (f(x + h) - f(x)) / h
        elif n == 2:
            return (-3 * f(x) + 4 * f(x + h) - f(x + 2 * h))/(2*h)
    elif type == 'b':
        if n == 1:
            return (f(x) - f(x - h))/h
        elif n == 2:
            return (3 * f(x) - 4*f(x - h) + f(x - 2 * h))/(2 * h)
    elif type == 'c':
        if n == 1:
            return (f(x + h) - f(x - h))/(2 * h)
        elif n == 2:
            return (f(x - 2*h) - 8*f(x - h) + 8*f(x + h) - f(x + 2*h))/(12*h)
        
def plot_error(x0):
    h = np.logspace(-8, 0, 9)
    type_list = ['f', 'b', 'c']
    order = [1, 2]
    for type in type_list:
        for n in order:
            error = abs(approx_fp(f, x0, h, type, n) - f_prime(np.array([x0])))
            if type == 'c':
                plt.loglog(h, error, label = 'Order{} {}'.format(2*n, type.upper()),\
                           marker = 'o')
            else:
#                error = abs(approx_fp(f, x0, h, type, n) - f_prime(np.array([x0])))
                plt.loglog(h, error, label = 'Order{} {}'.format(n, type.upper()),\
                           marker = 'o')
    plt.legend(loc = 'upper left')
    plt.xlabel('h')
    plt.ylabel('Absolute Error')
    plt.show()
    plt.close(0)
    
def diff_high(f, x0, h):
    n = x0.shape[0]
    base = []
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1
        base.append(e)
    cols = []
    for e in base:
        cols.append((f(x0 + h*e) - f(x0 - h*e))/(2 * h))
    jacob = np.column_stack(cols)
    return jacob
    
def test_func(x_vec):
    x, y = x_vec[0], x_vec[1]
    return np.array([x**2, x**3 - y])

def compare_method(x0, h):
    # compute using sympy
    print('Time Sympy Method')
    qe.util.tic()
    symp_result = g_prime(x0)
    qe.util.toc()
    
    #compute using approximation
    print('Time Approximation Method')
    qe.util.tic()
    approx_result = approx_fp(g, x0, h, 'c', 1)
    qe.util.toc()
    
    #compute using autograd
    print('Time Autograd Method')
    g2 = lambda x: anp.log((anp.sin(x**0.5))**0.5)
    g_grad = autograd.grad(g2)
    qe.util.tic()
    anp_result = g_grad(x0)
    qe.util.toc()
    
    return symp_result, approx_result, anp_result


def taylor_sin(x):
    result = 0
    cur_term = 10 * anp.abs(x)
    track_list = [1]
    i = 0
    while anp.abs(track_list[-1]) >= 1e-10:
        if (i + 1) % 4:
            cur_term = (i % 2) * x**i/np.math.factorial(i)
        else:
            cur_term = (i % 2) * (-1) * x**i/np.math.factorial(i)
#        if i % 2 == 0:
#            cur_term = 0
#        elif (i + 1) % 4 == 0:
#            cur_term = -x**i/np.math.factorial(i)
#        else:
#            cur_term = x**i/np.math.factorial(i)
        if cur_term != 0:
            track_list.append(cur_term)
        result += cur_term
        i += 1
    return result
    
def plot_sin_der():
    sin_first = autograd.grad(taylor_sin)
    sin_second = autograd.grad(sin_first)
    x_range = np.linspace(-np.pi, np.pi, 1000)
    sin_first_list = []
    sin_sec_list = []
    for x in x_range:
        sin_first_list.append(sin_first(x))
        sin_sec_list.append(sin_second(x))
    plt.plot(x_range, sin_first_list)
    plt.plot(x_range, sin_sec_list)
    plt.plot(x_range, np.sin(x_range))
    plt.show()
    
#def f(x_vec):
#    x, y = x_vec[0], x_vec[1]
#    return np.array([np.e**x * np.sin(y) + y**3, 3*y - np.cos(x)])

def f_auto(x_vec):
    x, y = x_vec[0], x_vec[1]
    return np.array([anp.e**x * anp.sin(y) + y**3, 3*y - anp.cos(x)])

def compare_jacob(x0):
    # compute using sympy
    f2x_exp = sy.diff(sy.log((sy.sin(x**0.5))**0.5), x)
    f2x_prime = lambda y: sy.N(f_prime_exp.subs(x, y))
    print('Time Sympy Method')
    qe.util.tic()
    symp_x = g_prime(x0)
    qe.util.toc()
    
    #computee using approxmation
    print('Time Approximation Method')
    qe.util.tic()
    approx_result = diff_high(f_auto, x0, 1e-5)
    qe.util.toc()
    
    #compute using autograd
    f_jacob = autograd.jacobian(f_auto)
    qe.util.tic()
    anp_result = f_jacob(x0)
    qe.util.toc()
    
    return approx_result, anp_result
    

if __name__=='__main__':
    plot_f()
    
    plot_error(1.0)
    
    compare_method(np.pi/4, 1e-5)
    
    plot_sin_der()
    
    compare_jacob(np.array([1.0,1.0]))