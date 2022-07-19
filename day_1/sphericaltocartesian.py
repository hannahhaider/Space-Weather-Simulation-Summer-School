#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" A 3D plot script for spherical coordinates
Created on Mon Jul 18 15:16:35 2022"""

__author__ : 'hannah haider'
__email__ : 'hzhaider@ucsd.edu'

import numpy as np
from math import pi 
import matplotlib.pyplot as plt

def spherical_to_cartesian(r,phi,theta):
    """Inputs r, phi, and theta are converted to cartesian coordinates x,y,z using the following conversion 
    """
    coordinates = { 'x': (r*np.sin(phi)*np.cos(theta) ), 
                    'y':(r*np.sin(phi)*np.sin(theta)), 
                    'z': (r*np.cos(phi))
                   }
    return coordinates

if __name__ == '__main__':
    fig = plt.figure() #better control
    axes= fig.gca(projection = '3d') # make 3d axes
    r = np.linspace(0,1)
    theta = np.linspace(0,2*np.pi)
    phi = np.linspace(0,2*np.pi)
    cartesian_coords = spherical_to_cartesian(r, phi, theta) 
    axes.plot(cartesian_coords['x'], cartesian_coords['y'],cartesian_coords['z'])