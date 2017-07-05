#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 09:53:05 2017

@author: luxihan
"""

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import pandas as pd
from functools import reduce
from matplotlib.colors import LogNorm

def plot_ans():
    ans_data = np.load('anscombe.npy')
    section1 = ans_data[:, [0, 1]]
    section2 = ans_data[:, [2, 3]]
    section3 = ans_data[:, [4, 5]]
    section4 = ans_data[:, [6, 7]]
    dom = np.linspace(3, 16, 200)
    
    data_list = [section1, section2, section3, section4]
    for i, sec in enumerate(data_list):
        plt.subplot(2, 2, i + 1)
        plt.scatter(sec[:, 0], sec[:, 1])
        plt.plot(dom, 1/2 * dom + 3, label = 'Regression line')
        plt.title('Section {}'.format(i + 1))
    plt.tight_layout()
    plt.show()
    
def compute_bbp(n, v, dom):
    mult = lambda x, y:x * y
    if n == v or v == 0:
        choice = 1
    else:
        choice = reduce(mult, np.linspace(1, n, n)) / \
                   (reduce(mult, np.linspace(1, v , v)) *\
                    reduce(mult, np.linspace(1, n - v, n - v)))
    rv = choice * dom ** v * dom **(n - v)
    return rv

    
def plot_bbp():
    dom = np.linspace(0, 1, 200)
    n_list = [0, 1, 2, 3]
    sub_plot_num = 0
    
    plt.figure(figsize = (6, 10))
    for n in n_list:
        for v in range(n + 1):
            sub_plot_num += 1
            bbp_val = compute_bbp(n, v, dom)
            plt.subplot(5, 2, sub_plot_num)
            plt.plot(dom, bbp_val, label = r'n = {}, v = {}'.format(n, v + 1))
            plt.tick_params(which = ' both', top = 'off', right = 'off')
            plt.xlim((0, 1))
            plt.ylim((0, 1))
            plt.legend(loc='upper left')
            
    plt.show()
    
def plot_mlb():
    mlb_array = np.load('MLB.npy')
    mlb_data = pd.DataFrame(mlb_array, columns = ['weight', 'height', \
                                                  'age'])
    pd.tools.plotting.scatter_matrix(mlb_data, alpha=0.2)
    plt.show()
    
def plot_earthquake():
    year, magnitude, latitude, longitude = np.load('earthquakes.npy').T
    eq_data = pd.DataFrame(np.array([year, magnitude, latitude, longitude]).T,\
                            columns = ['year', 'magnitude', 'latitude', 'longitude'])
    
    #histogram of how many
    eq_data['year'] = eq_data['year']//1
    agg_data = eq_data.groupby('year').size().rename('count')
    agg_data = agg_data.reset_index()
    plt.plot(agg_data['year'], agg_data['count'], marker = 'D')
    plt.xlabel('year')
    plt.ylabel('Count')
    plt.title('Number of Earthquakes by Year')
    plt.show()
    
    #Histogram of magnitude
    plt.hist(eq_data['magnitude'])
    plt.xlabel('Magnitude')
    plt.ylabel('Count')
    plt.title('Histogram of Earthquakes of Different Magnitude')
    plt.show()
    
    #Plot the scatter plot of the location of earthquake
    plt.scatter(eq_data['latitude'], eq_data['longitude'], \
                s = (eq_data['magnitude'] - min(eq_data['magnitude'])) * 50, \
                    c = eq_data['magnitude'],\
                    alpha = 0.7)
    cbar = plt.colorbar()
    cbar.set_label('Magnitude')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.show()
    
def plot_rbf():
    x = np.linspace(-3, 3, 1000)
    y = np.linspace(-3, 3, 1000)
    X, Y = np.meshgrid(x, y)
    
    Z = (1 - X)**2 + 100 * (Y - X**2)**2
    
    #plot the heat map
    plt.subplot(1, 2, 1)
    plt.pcolormesh(X, Y, Z, cmap = 'viridis')
    plt.colorbar()
    
    #plot the countour map
    plt.subplot(1, 2, 2)
    plt.contour(X, Y, Z, 100, cmap = 'viridis')
    plt.colorbar()
    plt.show()
    
    
def plot_countries():
    country = np.load('countries.npy')
    pop, gdp,hma, hfe = country.T
    gdp_per = gdp / pop
    
    #plot the histogram of gdp
    index = np.argsort(gdp_per)
    position = np.arange(1, len(gdp) + 1)
    plt.bar(position, gdp_per[index], align = 'center')
    label = ["Austria", "Bolivia", "Brazil", "China",
        "Finland", "Germany", "Hungary", "India",
        "Japan", "North Korea", "Montenegro", "Norway",
        "Peru", "South Korea", "Sri Lanka", "Switzerland",
        "Turkey", "United Kingdom", "United States", "Vietnam"]
    
    plt.xticks(position, np.array(label)[index], rotation = 90)
    plt.ylabel('GDP per capita')
    plt.tight_layout()
    plt.show()
    
    #plot the correlation between gdp and population
    plt.scatter(pop, gdp)
    plt.xlabel('Population')
    plt.ylabel('GDP')
    plt.title('GDP-Popluation Scatter Plot')
    plt.show()
    
    #plot the histogram of male's height
    plt.hist(hma)
    plt.xlabel('Male Height')
    plt.title('Histogram of Male Height')
    plt.show()
    
    
    