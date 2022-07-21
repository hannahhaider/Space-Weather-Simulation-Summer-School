#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 11:11:39 2022

@author: hannahhaider
"""
__author__ = 'Hannah Haider'
__email__ = 'hzhaider@ucsd.edu'
#%%making a pcolormesh plot of the total electron content
#dataset = nc.Dataset('/Users/hannahhaider/Documents/Data/wfs.t12z.ipe05.20220721_140000.nc')

import matplotlib.pyplot as plt
import netCDF4 as nc 
import numpy as np 
import argparse 

#modifying code to save as many as you want 
def parse_args():
    
    """function to parse input arguments"""
    parser = argparse.ArgumentParser(description = \
                                     'This code reads a netCDF4 file and plots the Total Electron Content as a pcolormesh plot, saving it as a png file with the same filename ')
    
    parser.add_argument('-file_in', nargs = '+', \
                        help = 'Input the filename you want to read', \
                            type = str)
    


    args = parser.parse_args()
    
    return args

#filename = '/Users/hannahhaider/Documents/Data/wfs.t12z.ipe05.20220721_140000.nc'


def plot_tec(dataset,figsize = (12,6)):
    """this function plots the total electron content from a NetCD4 file in a pcolormesh plot."""

    fig, ax = plt.subplots(1, figsize = figsize)
    
    longitude = dataset['lon'][:]
    latitude = dataset['lat'][:]
    
    tec = dataset['tec'][:] #the color index 
    
    colors = ax.pcolormesh(longitude,latitude,tec)
    fig.colorbar(colors,ax=ax, label=dataset['tec'].units)
    
    ax.set_title('Total Electron Content', fontsize = 18)
    ax.set_xlabel('Longitude (degrees)', fontsize = 16)
    ax.set_ylabel('Latitude (degrees)', fontsize = 16)

    
    return fig, ax



 # Will only run if this is run from command line as opposed to imported
if __name__ == '__main__':  # main code block

    args = parse_args()
    print(args)
    
    for file in args.file_in:      
        
        filename = file
        print(filename)
    
        dataset = nc.Dataset(filename) #obtain filename
        fig, ax = plot_tec(dataset) #save output of plot_tec function as fig, ax
        outfile= filename + '.png' #name outfile
        fig.savefig(outfile) #save the figure as outfile
        plt.show() #show the figure 

    