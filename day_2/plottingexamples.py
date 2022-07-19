#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 15:32:25 2022

@author: hannahhaider
"""

import numpy as np 
import datetime as dt 
import matplotlib.pyplot as plt 
import argparse 


def parse_args():
    
    """function to parse input arguments"""
    parser = argparse.ArgumentParser(description = \
                                     'This code reads an ascii file, storing the time, year, day, hour, minute, and SYM/H data in a dictionary. It also saves a plot of the time versus SYM/H data as an outfile. ')
    
    parser.add_argument('-file_in', \
                        help = 'Input the filename you want to read', \
                            type = str)
    
    parser.add_argument('-file_out', \
                        help = 'This is the name of the output file you would like to save', \
                            type = str)
    
    args = parser.parse_args()
    
    return args 

def read_ascii_file(filename, index):
    """This function opens and reads an ascii file, storing it in a data dictionary, then plots the specified [indexed] SYM/H data versus time"""
    
    with open(filename) as f:
        data_dictionary = {'time':[],
                           'year':[],
                           'day':[],
                           'hour':[],
                           'minute':[],
                           'symh':[]
                          }  #creating a dictionary for our data      
        for line in f:
            tmp = line.split()
            data_dictionary["year"].append(int(tmp[0]))
            data_dictionary["day"].append(int(tmp[1]))
            data_dictionary["hour"].append(int(tmp[2]))
            data_dictionary["minute"].append(int(tmp[3]))
            data_dictionary["symh"].append(int(tmp[index]))
            
            #create datetime in each line
            time0 = dt.datetime(int(tmp[0]),1,1,int(tmp[2]),int(tmp[3]),0) + dt.timedelta(days=int(tmp[1])-1)
            data_dictionary["time"].append(time0)
        
    return data_dictionary
# Will only run if this is run from command line as opposed to imported
if __name__ == '__main__':  # main code block
    
    args = parse_args()
    print(args)
    
    filename = args.file_in
    print(filename)
    
    outfile= args.file_out
    print(outfile)


data = read_ascii_file(filename, index=-1)

time = np.array(data["time"])
symh = np.array(data["symh"])

fig,ax = plt.subplots()

max_data = np.argmax(symh)
min_data = np.argmin(symh)
#print(max_data)
#print(min_data)

minsymh = min(symh)
time_of_minsymh = time[min_data].isoformat()
print("The time that the minimum SYM/H occurs is ",time_of_minsymh)


# add a operator on array, lp is a bool numpy array
lp = symh < -100
#print(lp)

#pass lp to a subscript operator [], returning a new array containing elements in the operand
ax.plot(time[lp], symh[lp], marker ='+', 
        linestyle = '', 
        c = 'tab:orange',
        label = '<-100 nT',
        alpha = 0.6)


ax.plot(time, symh, marker ='.', c = 'gray', label = 'All events', alpha = 0.5)
ax.axvline(time[max_data]) #plotting vertical line at max of symh
ax.axvline(time[min_data]) #plotting vertical line at min of symh 
ax.set_xlabel('Year of 2013')
ax.set_ylabel('SYMH (nT)')
ax.grid(True)
ax.legend()
#outfile = 'plot_example1.png'
#print('Writing File: ' + outfile)
plt.savefig(outfile)
plt.close()

