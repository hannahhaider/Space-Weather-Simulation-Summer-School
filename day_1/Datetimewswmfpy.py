#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 15:06:38 2022

@author: hannahhaider
"""
import numpy as np
from datetime import datetime
from swmfpy.web import get_omni_data

start_time = datetime(2001, 7, 14)
end_time = datetime(2001, 7, 15)
data = get_omni_data(start_time, end_time) 
data.keys()

import matplotlib.pyplot as plt 
al = data['al']
plt.plot(al)