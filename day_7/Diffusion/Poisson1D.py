#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:19:06 2022

@author: hannahhaider
"""

"""
Solution of a 1D Poisson equation: -u_xx = f
Domain: [0,1]
BC: u(0) = u(1) = 0
with f = (3*x + x^2)*exp(x)
Analytical solution: -x*(x-1)*exp(x)
Finite differences (FD) discretization: second-order diffusion operator
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation    #We have to load this
from math import pi
#matplotlib qt
plt.close()

"Specifying order of approximation"
order = [1,2]
order = order[1]
#flag= input("Specify the order of approximation, first or second: ") #have the user choose first or second order approximation

#if flag == "first":
   # order = order[0] #1st order approximation
#else:
    #order = order[1] #2nd order approximation

"Number of points"
N = 16
Dx = 1/N
x = np.linspace(0,1,N+1)
xC = np.linspace(0, 1+ Dx,N+2)
#xC = np.append(xC[-1],x)

"System matrix and RHS term"
A = (1/Dx**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1)) #initializing matrix 
AC = (1/Dx**2)*(2*np.diag(np.ones(N+2)) - np.diag(np.ones(N+1),-1) - np.diag(np.ones(N+1),1)) #initializing matrix for central difference
#F = (3*x + x**2)*np.exp(x) #treats all the points as interior points 
#F = np.concatenate(([0],F, [0])) #my attempt at forcing the boundary conditions
F = 2*(2*x**2 + 5*x -2)*np.exp(x) #RHS
FC = 2*(2*xC**2 + 5*xC -2)*np.exp(xC) #RHS for central difference

"explicit boundary conditions:"
#x = 0
A[0,:] = np.concatenate(([1], np.zeros(N))) #adding row of 1, zeros
AC[0,:] = np.concatenate(([1], np.zeros(N+1))) #adding row of 1, zeros for central difference
FC[0] = 0
F[0] = 0

#x = 1
if order < 2: #if order is first-order approximation
    A[N,:] = np.concatenate((np.zeros(N-1), [-1/Dx], [1/Dx])) #enforcing Neumann Boundary conditions using first order approximation 
else: #if the order is second-order approximation
    A[N,:] = (1/Dx)*np.concatenate((np.zeros(N-2), [1/2], [-2], [3/2])) #Neumann boundary condition with 2nd order approximation 
    AC[N+1,:]= (1/(2*Dx))*np.concatenate((np.zeros(N-1),[-1,0,1])) #centered difference approximation of the derivative 
    FC[N+1]=0 #central RHS BC forced
F[N] = 0 #RHS BC forced


"Solution of the linear system AU=F"
U = np.linalg.solve(A,F) #solution vector for 1st and 2nd order approx
UC= np.linalg.solve(AC, FC) #solution vector for central diff
#ua = -x*(x-1)*np.exp(x)
ua = 2*x*(3-2*x)*np.exp(x) 
uac = 2*xC*(3-2*xC)*np.exp(xC)

"Plotting solution"
plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
plt.plot(x,U,':ob',linewidth=2,label='$\widehat{u}$')
plt.plot(xC, UC, ':og', linewidth = 2, label = '$\widehat{u}$ central')
plt.legend(fontsize=12,loc='upper left')
plt.grid()
plt.axis([0,1,0, 6])
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)

"Compute error"
error_c = np.max(np.abs(UC-uac)) #computing central error
error = np.max(np.abs(U-ua)) #computing error between first/second and solution
print("Linf error u: %g\n" % error)
print("Central error u: %g\n" % error_c)
