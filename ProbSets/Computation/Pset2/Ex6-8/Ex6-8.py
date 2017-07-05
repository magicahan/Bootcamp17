#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 18:54:31 2016

@author: luxihan
"""

import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
import os

graph = True
####################
######Ex 1  ########
####################

# retrive the dictioary for date and days
base = datetime.datetime(2015, 9, 21)   # use 2015 - 2016 as the leap year base
num_days = 366
dates = [base + datetime.timedelta(days=i) for i in range(num_days)]
dates = pd.DataFrame({"date":dates})
dates["day"] = dates.index + 1
dates["date"] = dates["date"].apply(lambda x: x.strftime("%m%d"))
# set a dictionary with the format {date(eg. 12-25): day(integer between 1-366)}
date_dict = dates.set_index("date").to_dict()['day']

# set the critical date of lifetime
date1 = datetime.datetime(1975, 1, 22)
date2 = datetime.datetime(1980, 8, 1)
date3 = datetime.datetime(1988, 7 ,14)
date4 = datetime.datetime(1993, 8, 17)
date5 = datetime.datetime(1998, 5, 5)
date6 = datetime.datetime(2003, 5, 15)
date7 = datetime.datetime(2006, 6, 3)
date8 = datetime.datetime(2010, 10, 1)
date9 = datetime.datetime(2016, 10, 31)

day_born = date_dict[date1.strftime("%m%d")]
day_little = date_dict[date3.strftime("%m%d")]

# build a list of date and filename and location name to loop over
date_list = [date1, date2, date4, date5, date7, date9]
file_list = ["indianapolis.csv", "pittsburgh.csv", "miami.csv", "WashingtonDC.csv", "chicago.csv"]
location = ["ind", "pit", "mia", "dc", "chi"]
df = pd.DataFrame()

for i in range(len(date_list) - 1):
    df_temp = pd.read_csv(file_list[i], usecols = ["DATE", "TMAX", "TMIN"])
    # get rid of the extreme outliers
    df_temp = df_temp[(df_temp.TMAX < 200) & (df_temp.TMAX > -200)]
    df_temp = df_temp[(df_temp.TMIN < 200) & (df_temp.TMIN > -200)]
    # get the average of the temperature for all of the stations
    df_temp = df_temp.groupby("DATE").mean().reset_index()
    df_temp["DATE"] = pd.to_datetime(df_temp["DATE"], format = "%Y%m%d")
    # convert all of the data veriables to month-day format and then convert
    # them to day number from 1- 366
    df_temp["date"] = df_temp["DATE"].apply(lambda x: x.strftime("%m%d"))
    df_temp["day"] = df_temp["date"].apply(lambda x: date_dict[x])
    df_temp["place"] = location[i]
    if i != 5:
        df_temp = df_temp[(df_temp.DATE >= date_list[i]) 
                            & (df_temp.DATE < date_list[i + 1])]
    else:
        df_temp = df_temp[(df_temp.DATE >= date_list[i]) 
                            & (df_temp.DATE <= date_list[i + 1])]
    df = df.append(df_temp, ignore_index = True)

    
if graph:
    '''
    ---------------------------------------------------------------------------
    fig, ax     =  plot object
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved
    ---------------------------------------------------------------------------
    '''
    fig = plt.figure(figsize = (15, 6))
    fig.suptitle('Lifetime Temperature', fontsize=18, fontweight='bold')
    ax = fig.add_subplot(1, 1, 1)
    
    ax.scatter(df[df.place != "chi"].day, df[df.place != "chi"].TMAX, c = "black", s = 7.5, alpha = 0.5)
    ax.scatter(df[df.place == "chi"].day, df[df.place == "chi"].TMAX, c = "maroon", linewidth = "0.5", s = 12.5, alpha = 0.8)
    ax.scatter(df[df.place != "chi"].day, df[df.place != "chi"].TMIN, c = "black", s = 7.5, alpha = 0.5)
    ax.scatter(df[df.place == "chi"].day, df[df.place == "chi"].TMIN, c = "maroon", linewidth = "0.5", s = 12.5, alpha = 0.8)
    label1 = "TMAX when Born"
    label2 = "TMAX when won the Regional Championship"
    label3 = "TMIN when Born"
    label4 = "TMIN when won the Regional Championship"
    ax.scatter(day_born, df[df.DATE == date1].TMAX, marker = "D", c = "yellow", s = 50)
    ax.annotate(label1, xy=(day_born, df.loc[df.DATE == date1, "TMAX"] + 5),
                    xytext=(day_born, df.loc[df.DATE == date1, "TMAX"] + 60),
                    arrowprops=dict(facecolor='black'),
                    horizontalalignment='center', verticalalignment='top')
    ax.scatter(day_little, df[df.DATE == date3].TMAX, marker = "D", c = "yellow", s = 50)
    ax.annotate(label2, xy=(day_little, df.loc[df.DATE == date3, "TMAX"] + 2),
                    xytext=(day_little, df.loc[df.DATE == date3, "TMAX"] + 20),
                    arrowprops=dict(facecolor='black'),
                    horizontalalignment='center', verticalalignment='bottom')
    ax.scatter(day_born, df[df.DATE == date1].TMIN, marker = "D", c = "yellow", s = 50)
    ax.annotate(label3, xy=(day_born, df.loc[df.DATE == date1, "TMIN"] - 5),
                    xytext=(day_born, df.loc[df.DATE == date1, "TMIN"] - 60),
                    arrowprops=dict(facecolor='black'),
                    horizontalalignment='center', verticalalignment='bottom')
    ax.scatter(day_little, df[df.DATE == date3].TMIN, marker = "D", c = "yellow", s = 50)
    ax.annotate(label4, xy=(day_little, df.loc[df.DATE == date3, "TMIN"] - 5),
                    xytext=(day_little, df.loc[df.DATE == date3, "TMIN"] - 50),
                    arrowprops=dict(facecolor='black'),
                    horizontalalignment='center', verticalalignment='bottom')
    
    ax.xaxis.set_major_locator(ticker.FixedLocator([1, 91.5, 183, 274.5, 366]))
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_locator(ticker.FixedLocator([45.75, 137.25, 228.75, 320.25]))
    ax.xaxis.set_minor_formatter(ticker.FixedFormatter(["Autumn", "Winter", "Spring", "Summer"]))
    plt.gca().yaxis.grid(True)
    plt.xlabel("Period in One Year", fontsize = 15)
    plt.ylabel("Temperature in Farhenhite", fontsize = 15)
    plt.tick_params(which='minor', labelsize = 15)
    plt.xlim((0, 367))
    #plt.tight_layout()
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = "images"
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "temperature")
    plt.savefig(output_path)
    plt.close()


####################
######Ex 2  ########
####################
## Ex 2. a)
print("Ex 2. a)")
df_heart = pd.read_csv("lipids.csv", skiprows = [0, 1, 2], header = 0)
df_heart = df_heart[df_heart.diseased == 1]

if graph:
    '''
    ---------------------------------------------------------------------------
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved  
    weight      = np.array object; the weighting scheme for the frequency plotting
    ---------------------------------------------------------------------------
    '''
    # create a weight for the plot to get the frequency
    figt = plt.figure()
    weight = (1 / df_heart.shape[0]) * np.ones(len(df_heart))
    n_array, bin_cuts, patches = plt.hist(df_heart.chol, 25, weights = weight)
    plt.xticks(np.round(bin_cuts, 1), rotation=60, fontsize = 8.5)
    plt.xlabel(r"Cholestrol Level")
    plt.ylabel(r"Frequency")
    plt.title("Cholestrol Level Histogram")
    fig.set_tight_layout(True)
    plt.xlim((None, 420))
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = "images"
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "hist_2d")
    plt.savefig(output_path)
    plt.close()

# get the index for the highest frequency bin and get the midpoint
max_index = np.argmax(n_array, axis = 0)        # highest frequency bin
midpoint = (bin_cuts[max_index] + bin_cuts[max_index + 1]) / 2
print("    The midpoint of the last bin is {:.2f}".format(midpoint))            

## Ex 2. b)
if graph:
    '''
    ---------------------------------------------------------------------------
    fig, ax     = plot object
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved  
    chol        = np.array the level of cholestrol 
    trig        = np.array the level of Trigliceride
    hist        = number of observations of each bin
    xedges      = the tick for x axis
    yedges      = the tick for y axis
    x_midp      = mid point for each bin for x axis variable
    y_midp      = mid point for each bin for y axis variable
    elements    = number of points should be sepcified
    ---------------------------------------------------------------------------
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection ='3d')
    bin_num = 25
    chol = np.array(df_heart.chol)
    trig = np.array(df_heart.trig)
    hist, xedges, yedges = np.histogram2d(chol, trig, bins = bin_num)
    hist = hist / hist.sum()
    x_midp = xedges[: -1] + 0.5 * (xedges[1] - xedges[0])
    y_midp = yedges[: -1] + 0.5 * (yedges[1] - yedges[0])
    elements = (len(xedges) - 1) * (len(yedges) - 1)
    ypos, xpos = np.meshgrid(y_midp, x_midp)
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros(elements)
    dx = (xedges[1] - xedges[0]) * np.ones_like(bin_num)
    dy = (yedges[1] - yedges[0]) * np.ones_like(bin_num)
    dz = hist.flatten()
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='g', zsort='average')
    ax.set_xlabel("Cholestrol Level")
    ax.set_ylabel("Trigliceride Level")
    ax.set_zlabel("Frequency")
    plt.title("Heart Disease 3D Histogram")
    fig.set_tight_layout(True)
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = "images"
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "hist_3d")
    plt.savefig(output_path)
    plt.show()

####################
######Ex 3  ########
####################
df_job = pd.read_csv("payems.csv", skiprows = range(5))
df_job.date = pd.to_datetime(df_job.date, format = "%m/%d/%y")
# correct all of the years misspecified by .to_datetime() method
for i, row in df_job.iterrows():
    if row.date.year > 2016:
        df_job.iloc[i, 0] = datetime.datetime(row.date.year - 100, row.date.month, row.date.day)
df_job.sort_values(by = "date")
df_recession = pd.read_excel("NBERchronology.xlsx").iloc[:34, :]
df_recession = df_recession.iloc[-14 :, :].reset_index()
#df_recession["Peak month"] = pd.to_datetime(df_recession["Peak month"], fromat = "%M %Y")
for i, row in df_recession.iterrows():
    df_recession.loc[i, "Peak month"] = datetime.datetime.strptime(row["Peak month"], "%B %Y")
    
# create the list to sotre all of the inputs needed to plot the graph
# this is a list of tuple: (period, employment level, year of recession)
stat_list = []
# for 1929
per_list = [(i + 1) * 12 for i in range(8)]
ems_series = df_job.loc[: 7, "payems"]
# every element is weighted by the peak month number
ems_series = ems_series / ems_series[0]
rec_name = (df_recession.loc[df_recession["Peak month"]\
                         .apply(lambda x: x.year == 1929), "Peak month"].iloc[0])\
                         .strftime(format = "%Y-%m")
                         
stat_list.append((per_list, ems_series, rec_name))

# for 1937
rec_name = (df_recession.loc[df_recession["Peak month"]\
                         .apply(lambda x: x.year == 1937), "Peak month"].iloc[0])\
                         .strftime(format = "%Y-%m")
# create per_list (1937 recession containing missing data)
per_list = [0, 12, 24]
temp_list = [i + 24 + 1 for i in range(12 * 6)[5 : ]]
per_list += temp_list
end_date = datetime.datetime(1937 + 7, 7, 1)
end = df_job[df_job["date"].apply(lambda x: x == end_date)].index[0]
ems_series = df_job.loc[7 : end, "payems"]
ems_series = ems_series / ems_series.iloc[1]
stat_list.append((per_list, ems_series, rec_name))


per_list = list(range(12 * 8 + 1))
for i, row in df_recession.iloc[2 :, :].iterrows():
    # get the peak month as the base index
    base = df_job[df_job.date == row["Peak month"]].index[0]
    # extract the -1 year - 7 year
    df_sub = df_job.iloc[base - 12 : base + 12 * 7 + 1, :]
    ems_series = df_sub.payems
    ems_series = ems_series / ems_series.iloc[12]
    rec_name = row["Peak month"].strftime(format = "%Y-%m")
    stat_list.append((per_list, ems_series, rec_name))
ref_level = np.ones_like(per_list)
    
if graph:
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)
    fig.suptitle('Recession Jobs Spider Plot', fontsize=14, fontweight='bold')
    for per, array, rec_name in stat_list:
        if "1929" in rec_name:
            rec_name += " Recession"
            plt.plot(per, array, label = rec_name, c= 'black', lw = 3)
        elif "2007" in rec_name:
            rec_name += " Recession"
            plt.plot(per, array, label = rec_name, c = 'red', lw = 3)
        elif "1945" in rec_name:
            plt.plot(per, array, label = rec_name, ls = ":", lw = 3)
        else:
            rec_name += " Recession"
            plt.plot(per, array, label = rec_name)
    plt.plot(per_list, ref_level, c = 'grey', linestyle = "--", lw = 1.5)
    plt.legend(loc = "upper left")
    tick = [i * 12 for i in range(9)]
    tick_text = ["-1yr", "peak", "+1yr", "+2yr", "+3yr", "+4yr", "+5yr", "+6yr", "+7yr"] 
    plt.xticks(tick, tick_text)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.xlabel(r"Time from peak")
    plt.ylabel(r"Jobs/peak")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.tight_layout()
    plt.xlim((0, 97))
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = "images"
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "recession")
    plt.savefig(output_path,bbox_inches='tight')
    plt.close()
    
    
