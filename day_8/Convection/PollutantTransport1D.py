#!/usr/bin/env python
"""
Advection of a pollutant subject to a constant velocity

1D Convection-Diffusion equation: u_t -nu*u_xx + c*u_x = f
Domain: [0,1]
BC: u'(0) = 0, u(1) = u0
with f = 100*exp(-((x-0.8)/0.01)^2)*((sin(2*pi*t) + abs(sin(2*pi*t)))/2)

Finite differences (FD) discretization:
    - Second-order cntered differences advection scheme
    - First-order upwind
    - Limiters to switch from high to low-resolution
    
    
Tasks:
    - See what happens as we change time-step
    - See what happens as we change viscosity
    
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
nu = 0
c = -2
u0 = 0

"Scheme parameters"
beta = 0

"Number of points"
N = 32
Dx = 1/N
x = np.linspace(0,1,N+1)
xN = np.concatenate(([x[0]-Dx],x))
x_2 = np.linspace(0, 1+ Dx, N+2) #introducing a ghost state 


"Time parameters"
dt = 1/50
time = np.arange(0,3+dt,dt)
nt = np.size(time)

"Initialize solution variable"
U = np.zeros((N+2,nt))


for it in range(nt-1):

    "System matrix and RHS term"
    "Diffusion term"
    Diff = nu*(1/Dx**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1))
    Diff2 = nu*(1/Dx**2)*(2*np.diag(np.ones(N+2)) - np.diag(np.ones(N+1),-1) - np.diag(np.ones(N+1),1))
    "Advection term:"
        
    "Sensor"
    U0 = U[:,it]
    #uaux = np.concatenate(([U0[0]], U0,[U0[N]]))
    #Du = uaux[1:N+3] - uaux[0:N+2] + 1e-8
    #r = Du[0:N+1]/Du[1:N+2]
    
    "Limiter"
    #if beta>0:
        #phi = np.minimum(np.minimum(beta*r,1),np.minimum(r,beta))
        #phi = np.maximum(0,phi)
    #else:
       # phi = 2*r/(r**2 + 1)
        
    #phim = phi[0:N]
    #phip = phi[1:N+1]
        
    
    "Upwind scheme"
    cp = np.max([c,0])
    cm = np.min([c,0])
    
    #Advp = cp*(np.diag(1-phi) - np.diag(1-phip,-1))
    #Advm = cm*(np.diag(1-phi) - np.diag(1-phim,1))
    #Alow = Advp-Advm
    order = 2
    if order < 2:
        "Advection term: first order upwind scheme by discretizing the Advection operator "
        c_plus = max(c,0)
        c_minus = min(c,0)
        Advp = c_plus*(np.diag(np.ones(N+1)) - (np.diag(np.ones(N), -1)))  #u_j - u_j-1, N-1 is the actual diagonal. N-2 is the upper or lower diagonal and the number after defines the upper/lower
        Advm = c_minus*(np.diag(np.ones(N+1))- (np.diag(np.ones(N), 1))) #u_j - u_j+1
        Adv = (1/Dx)*(Advp-Advm)
    else: 
        "Implementing second order upwind scheme "
        c_plus = max(c,0)
        c_minus = min(c,0)
        Advp = c_plus*(1.5*(np.diag(np.ones(N+1)))- (2*np.diag(np.ones(N),-1)) + 0.5*(np.diag(np.ones(N-1),-2)))
        Advm = c_minus*(1.5*(np.diag(np.ones(N+1)))-2*(np.diag(np.ones(N),1)) + 0.5*(np.diag(np.ones(N-1), 2)))
        
        Advp_2 = c_plus*(1.5*(np.diag(np.ones(N+2)))- (2*np.diag(np.ones(N+1),-1)) + 0.5*(np.diag(np.ones(N),-2))) #2nd order derivative, ghost state
        Advm_2 = c_minus*(1.5*(np.diag(np.ones(N+2)))-2*(np.diag(np.ones(N+1),1)) + 0.5*(np.diag(np.ones(N), 2)))
        
    Adv = (1/Dx)*(Advp-Advm)
    Adv_2 = (1/Dx)*(Advp_2 - Advm_2)
    
    "Centered differences"
    #Advp = -0.5*c*np.diag(phip,-1)
    #Advm = -0.5*c*np.diag(phim,1)
    #Ahigh = Advp-Advm
        
    #Adv = (1/Dx)*(Ahigh + Alow)
    A =  Adv + Diff
    A2 = Adv_2 + Diff2 
    
    "Source term"
    sine = np.sin(2*pi*time[it+1])
    sineplus = 0.5*(sine + np.abs(sine))
    F = 100*np.exp(-((x-0.8)/0.01)**2)*sineplus #introducing a pollutant 
    F2 = 100*np.exp(-((x_2-0.8)/0.01)**2)*sineplus 
    
    "Temporal terms"
    A = A + (1/dt)*np.diag(np.ones(N+1))
    A2 = A2 + (1/dt)*np.diag(np.ones(N+2))
    #F = F + U0/dt
    F2 = F2 + U0/dt

    "Boundary condition at x=0"
    A[0,:] = (1/Dx)*np.concatenate(([1.5, -2, 0.5],np.zeros(N-2)))
    F[0] = 0
    A2[0,:] = (1/Dx)*np.concatenate(([1.5, -2, 0.5],np.zeros(N-1)))
    F2[0] = 0
    "Boundary condition at x=1"
    A[N,:] = np.concatenate((np.zeros(N),[1]))
    F[N] = u0
    
    A2[N+1,:]=(1/(Dx**2))*np.concatenate((np.zeros(N-1),[1, -2, 1])) #ghost state, second order derivative approx
    F2[N+1] = 0 #forced RHS BC
    


    "Solution of the linear system AU=F"
    u = np.linalg.solve(A2,F2)
    U[:,it+1] = u
    u = u[0:N+1]


"Animation of the results"
fig = plt.figure()
ax = plt.axes(xlim =(0, 1),ylim =(u0-1e-2,u0+0.5)) 
myAnimation, = ax.plot([], [],':ob',linewidth=2)
plt.grid()
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)

def animate(i):
    
    u = U[0:N+1,i]
    plt.plot(x,u)
    myAnimation.set_data(x, u)
    return myAnimation,

anim = animation.FuncAnimation(fig,animate,frames=range(1,nt),blit=True,repeat=False)


if nu>0:
    "Peclet number"
    P = np.abs(c*Dx/nu)
    print("Pe number Pe=%g\n" % P);

"CFL number"
CFL = np.abs(c*dt/Dx)
print("CFL number CFL=%g\n" % CFL);



