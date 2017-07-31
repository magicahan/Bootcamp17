#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 21:37:48 2017

@author: luxihan
"""

import numpy as np

def steepest(Q, b, x_init, tol):
    x_now = x_init.copy()
    while (Q @ x_now) > tol:
        alpha = (x_now.T @ Q @ Q @ x_now) / (x_now.T @ Q @ Q @ Q @ x_now)
        x_now = x_now - alpha * Q @ x_now
    return x_now