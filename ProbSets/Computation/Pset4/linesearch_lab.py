#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 12:03:57 2017

@author: luxihan
"""
rho = 0.382

def golden_search(f, limits, niter):
    a, b = limits
    for i in niter:
        a_p = a + rho * (b - a)
        b_p = a + (1 - rho) * (b - a)
        if f(a_p) >= f(b_p):
            a = a_p
        else:
            b = b_p
    return (b - a) / 2

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

def bisection(f, limits, niter):
    a, b = limits
    for i in range(niter):
        mid = (a + b)/2
        if approx_fp(f, mid, 1e-5, 'c', 1) < 0:
            a = mid
        elif approx_fp(f, mid, 1e-5, 'c', 1) > 0:
            b = mid
        else:
            return mid
    return (a + b)/2

def newton(f, f_p, f_sp, x0):
    eps = 1e-6
    error = abs(x0) * eps + 10
    error_list = [error]
    x_now = x0
    while error >= abs(x_now) * eps:
        if error[-1] > error[-2]:
            raise ValueError('Not Converging')
        x_prev = x_now
        x_now = x_prev - f_p(x_prev)/f_sp(x_prev)
        error = abs(x_now - x_prev)
        error_list.append(error)
    return x_now
        
def secant(f, f_p, x0, x1):
    eps = 1e-6
    error = abs(x0) * eps + 10
    error_list = [error]
    x_now = x0
    while error >= abs(x_now) * eps:
        if error[-1] > error[-2]:
            raise ValueError('Not Converging')
        x2 = x1 - (x1 - x0)/(f_p(x1) - f_p(x0)) * f_p(x1)
        x0 = x1
        x1 = x2
        error = abs(x1 - x0)
        error_list.append(error)
    return x_now

def btls(f, f_p, xk, p, c, rho):
    alpha = 1
    while f(xk + alpha * p) <= f(xk) + c * alpha * f_p(xk) * p:
        alpha *= rho
    return alpha * p
    