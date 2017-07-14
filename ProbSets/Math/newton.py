#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 16:50:01 2017

@author: luxihan
"""

def newton(x0, eps, f_p, f_dp):
    error = abs(x0) * eps + 10
    error_list = [error]
    x_now = x0
    while error >= abs(x_now) * eps:
        if error[-1] > error[-2]:
            raise ValueError('Not Converging')
        x_prev = x_now
        x_now = x_prev - f_p(x_prev)/f_dp(x_prev)
        error = abs(x_now - x_prev)
        error_list.append(error)
    return x_now