#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:35:48 2022

@author: hannahhaider
"""
from datetime import datetime
from swmfpy.web import get_omni_data

start_time = datetime(2001, 7, 14)
end_time = datetime(2001, 7, 15)
data = get_omni_data(start_time, end_time) 
data.keys()