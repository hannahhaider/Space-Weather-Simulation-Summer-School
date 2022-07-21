#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 10:27:09 2022

@author: hannahhaider
"""
#%% zip lists
names = ['Ahmed', 'Becky', 'Cantor']

ages = [21, 30, 45]

favorite_colors = ['Pink', 'Grey', 'Blue']

print(list(zip(names,ages,favorite_colors)))
[('Ahmed', 21, 'Pink'), ('Becky', 30, 'Grey'), ('Cantor', 45, 'Blue')]

for name, age, color in zip(names, ages, favorite_colors):
    print(name,age,color)
    
#%% a great example
from datetime import datetime 
num_of_days = 10
years = [2009]*num_of_days
months = [12]*num_of_days
days = list(range(1, 11))

times = [datetime(year, month, day) #temp variables
         for year, month, day
         in zip(years,months, days)] #list comprehension
for time in times:
    print(time.isoformat())
    
#%% pcolormesh plots
#it's like a plt.contourf plot but with needing to make a np.meshgrid()

import numpy as np
import matplotlib.pyplot as plt

num_of_x = 10
num_of_y = 20
x = np.linspace(0,1, num_of_x)
y = np.linspace(0,1,num_of_y)
z = np.random.randn(num_of_y, num_of_x) #the colors 
plt.pcolormesh(x,y,z)
plt.colorbar()

#%% NETCDF
#lots of space data in netcdf

import netCDF4 as nc 
dataset = nc.Dataset('/Users/hannahhaider/Documents/Data/wfs.t12z.ipe05.20220721_140000.nc')
print(dataset)

dataset['tec'][:] #how you get the numpy array of the data
dataset['tec'].units #how you get the units of the data
#%%making a pcolormesh plot of the total electron content

def plot_tec(dataset, figsize = (12,6)):
    fig, ax = 