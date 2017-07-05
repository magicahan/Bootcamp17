#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 08:29:51 2017

@author: luxihan
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def gen_rand(n):
    rand_mat = np.random.randn(n, n)
    mean_mat = np.mean(rand_mat, axis = 1)
    var_mat = np.var(rand_mat, axis = 1)
    return rand+mat, mean_amt, var_mat

def plot_rand(n_array):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(2, 1, 1)
    ax2 = fig1.add_subplot(2, 1, 2)
    fig2 = plt.figure()
    ax3 = fig2.add_subplot(2, 1, 1)
    a4 = fig2.add_subplot(2, 1, 2)
    for n in n_array:
        rand_mat, mean_mat, var_mat = gen_rand(n)
        ax1.plot(mean_mat, label = 'n = {}'.format(n))
        ax2.plot(var_mat)
        
#Problem 2
def gen_plot_tri(n):
    dom = np.linspace(-2*np.pi, 2 * np.pi, 200)
    fig = plt.figure(figsize = (5, 6))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)  
    #generate sin
    ax1.plot(np.sin(dom))
    ax1.set_title('Sin Function')
    
    #generate cosine
    ax2.plot(np.cos(dom))
    ax2.set_title('Cos Funciton')
    
    #generate arctan
    ax3.plot(np.arctan(dom))
    ax3.set_title('arctan Funciton')
    
    plt.suptitle('Functions')
    plt.tight_layout()
    plt.show()    
    return


def plot_discon():
    dom1 = np.linspace(-2, 0.99, 100)
    dom2 = np.linspace(1.01, 6, 100)
    
    plt.plot(dom1, 1/(dom1 - 1), lw = 2)
    plt.plot(dom2, 1/(dom2 - 1), lw = 2)
    plt.title(r'Function $1/{x-1}$')
    plt.ylim((-6, 6))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
def gen_tri_subplot():
    dom = np.linspace(0, 2 * np.pi, 200)
    fig = plt.figure(figsize = (7, 3.5))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)  
    ax4 = fig.add_subplot(2, 2, 4)
    #generate sin
    ax1.plot(np.sin(dom), color = 'g')
    ax1.set_title('sin(x)')
    
    #generate cosine
    ax2.plot(np.sin(2 * dom), color = 'r', linestyle = '--')
    ax2.set_title('sin(2x)')
    
    #generate arctan
    ax3.plot(2 * np.sin(dom), color = 'b', linestyle = '--')
    ax3.set_title('2 sin(x)')
    
    ax4.plot(2 * np.sin(2 * dom), color = 'm', linestyle = '--')
    ax4.set_title('2 Sin(2x)')
    
    plt.ylim((-2, 2))
    plt.xlim((0, 2 * np.pi))
    plt.suptitle('Functions')
    plt.tight_layout()
    plt.show()  
    
def gen_fars_plot():
    fars_array = np.load('FARS.npy')
    long = fars_array[:, 1]
    lat = fars_array[:, 2]

    #generate scatter plot
    plt.scatter(lat, long, color = 'black', marker = ',')
    plt.xlabel('Latitdue')
    plt.ylabel('Longitude')
    plt.title('Car Crash Locaiton')
    plt.axis('equal')
    plt.show()

    #generate histogram
    bins = np.linspace(0, 24, 25)
    plt.hist(fars_array[:, 0], bins = bins)
    plt.title('Car Crash Time Histogram')
    plt.show()
    
def gen_heat_contour():
    x = np.linspace(-2* np.pi, 2 * np.pi)
    y = np.linspace(-2* np.pi, 2 * np.pi)
    X, Y = np.meshgrid(x, y)
    
    Z = np.sin(X) * np.sin(Y) / (X * Y)
    
    #generate heat map
    plt.figure(figsize = (8,4))
    plt.subplot(1, 2, 1)
    plt.pcolormesh(X, Y, Z, cmap="viridis")
    plt.colorbar()
    plt.xlim((-2* np.pi, 2 * np.pi))
    plt.ylim((-2* np.pi, 2 * np.pi))
    
    #generate contour plot
    plt.subplot(1, 2, 2)
    plt.contour(X, Y, Z, 10, cmap="Spectral")
    plt.colorbar()
    plt.xlim((-2* np.pi, 2 * np.pi))
    plt.ylim((-2* np.pi, 2 * np.pi))
    plt.show()
    
    





