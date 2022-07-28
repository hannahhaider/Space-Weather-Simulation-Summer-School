#!/usr/bin/env python
"""
Solution of a 1D Convection-Diffusion equation: -nu*u_xx + c*u_x = f
Domain: [0,1]
BC: u(0) = u(1) = 0
with f = 1

Analytical solution: (1/c)*(x-((1-exp(c*x/nu))/(1-exp(c/nu))))

Finite differences (FD) discretization:
    - Second-order cntered differences advection scheme

"""
__author__ = 'Jordi Vila-Pérez'
__email__ = 'jvilap@mit.edu'


import numpy as np
import matplotlib.pyplot as plt
from math import pi
#%matplotlib qt
plt.close()
import matplotlib.animation as animation

"Flow parameters"
nu = 0.01
c = 2

"Number of points"
N = 32
Dx = 1/N
x = np.linspace(0,1,N+1)

"time parameters"
dt = 0.1
tf = 3
time = np.arange(0, tf+ dt, dt)
nt = np.size(time)

"Initializing solution of U"
U = np.zeros((N-1, nt)) #initializing solution vector 
#= U[:,0]  #initial condition 

for i in range(0, nt-1): 
    "System matrix and RHS term"

    "Diffusion term"
    Diff = nu*(1/Dx**2)*(2*np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),-1) - np.diag(np.ones(N-2),1))
        
    "Specifying order of approximation"
    order = 1
    if order < 2:
        "Advection term: first order upwind scheme by discretizing the Advection operator "
        c_plus = max(c,0)
        c_minus = min(c,0)
        Advp = c_plus*(np.diag(np.ones(N-1)) - (np.diag(np.ones(N-2), -1)))  #u_j - u_j-1, N-1 is the actual diagonal. N-2 is the upper or lower diagonal and the number after defines the upper/lower
        Advm = c_minus*(np.diag(np.ones(N-1))- (np.diag(np.ones(N-2), 1))) #u_j - u_j+1
        Adv = (1/Dx)*(Advp-Advm)
    else: 
        "Advection term: centered differences"
        Advp = -0.5*c*np.diag(np.ones(N-2),-1) #below the diagonal, u_j-1
        Advm = -0.5*c*np.diag(np.ones(N-2),1) #upper diagonal u_j+1
        Adv = (1/Dx)*(Advp-Advm)
        
    "Source term"
    F = np.ones(N-1)
        
    "Adding in temporal loop"
    A = Diff + Adv
    A = A + (1/dt)*(np.diag(np.ones(N-1))) #A = A + 1/dt*Identity matrix 
        
    F = F + (U[:,i])/dt
        
    u = np.linalg.solve(A, F) #solve system of equations 
    U[:, i+1] = u #loop through values
    #u = np.concatenate(([0], u, [0]))

"Solution of the linear system AU=F"
ua = (1/c)*(x-((1-np.exp(c*x/nu))/(1-np.exp(c/nu))))

"Animation of the results"
fig = plt.figure()
ax = plt.axes(xlim =(0, 1),ylim =(0,1/c)) 
plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
myAnimation, = ax.plot([], [],':ob',linewidth=2)
plt.grid()
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)

def animate(i):
    u = np.concatenate(([0],U[0:N+1,i],[0]))
    plt.plot(x,u)
    myAnimation.set_data(x, u)
    return myAnimation,
anim = animation.FuncAnimation(fig,animate,frames=range(1,nt),blit=True,repeat=False)


#plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
#plt.plot(x,u,':ob',linewidth=2,label='$\widehat{u}$')
#plt.legend(fontsize=12,loc='upper left')
#plt.grid()
#plt.axis([0, 1,0, 2/c])
#plt.xlabel("x",fontsize=16)
#plt.ylabel("u",fontsize=16)


"Compute error"
error = np.max(np.abs(u-ua))
print("Linf error u: %g\n" % error)

"Peclet number"
P = np.abs(c*Dx/nu)
print("Pe number Pe=%g\n" % P);


