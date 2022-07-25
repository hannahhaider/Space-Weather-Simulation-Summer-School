#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 09:44:22 2022

@author: hannahhaider
"""
#%% 
"""This script has two functions, one which calls the generic function (func(x)) 
cos(x) + xsin(x) where 'x' is the input and a second function, func_dot(x). 
It then plots the two outputs as f and fprime."""

import numpy as np
import matplotlib.pyplot as plt 

def func(input):
    """generic function"""
    return np.cos(input) + input*np.sin(input)

def func_dot(input):
    """derivative of generic function"""
    return  input*np.cos(input)


#specifying input argument
range_x_in = 6 #specifying beginning range of x
range_x_end = -range_x_in #specifying end range of x as symmetric
input_points = 1000 #specifying number of x points
x = np.linspace(range_x_in,range_x_end,input_points) #creating a linspace of x as the input

#calling functions to plot
f = func(x) #calling generic function
fprime = func_dot(x) #calling derivative function 


#plotting fprime and f on same plot
fig1 = plt.figure()
plt.grid()
plt.plot(x,f, c = 'gray')
plt.plot(x, fprime, c = 'blue')
plt.xlabel('x', fontsize = 16)
plt.legend([r'$y$', r'$dy/dt$'], fontsize = 16) #can also do r'$\dot y$', must be put in brackets like an array
plt.title('plotting a function, y, and the derivative', fontsize = 16)
#%%
#finite differences function definitions 
def forward_fin_diff(input, step_size):
    """approximates the derivative of f with a specified step size and input using forward finite differences 
    
    input: value which you are explicitly evaluating the function at.
    
    step_size: the discrete step size, or increment for the finite difference.
    """
    current_val = np.cos(input) + input*np.sin(input) # KO - you can reuse the function defined earlier
    following_val = np.cos(input + step_size) + (input+step_size)*np.sin(input + step_size)
    difference = (following_val - current_val)/step_size # + O(h)
    return difference

def backward_fin_diff(input, step_size):
    """approximates the derivative of f with a specified step size and input using backward finite differences. 
    
    input: value which you are explicitly evaluating the function at.
    
    step_size: the discrete step size, or increment for the finite difference.
    """
    current_val= np.cos(input) + input*np.sin(input)
    previous_val = np.cos(input-step_size) + (input-step_size)*np.sin(input-step_size)
    difference = (current_val - previous_val)/step_size #+ O(h)
    return difference

def central_fin_diff(input, step_size):
    """approximates the derivative of f with a specified step size and input using central finite differences.
    
    input: value which you are explicitly evaluating the function at.
    
    step_size: the discrete step size, or increment for the finite difference.
    """
    following_val = np.cos(input + step_size) + (input+step_size)*np.sin(input + step_size)
    previous_val = np.cos(input-step_size) + (input-step_size)*np.sin(input-step_size)
    difference = (following_val - previous_val)/(2*step_size) # + O(h^2)
    return difference

#initializing figure 
fig2 = plt.figure()
plt.plot(x, fprime, '-k')
plt.grid()
plt.xlabel(r'$x$')
plt.ylabel(r'$\dot y$')
plt.legend([r'$dot y$ truth'])

#initializing variables for step size, range of input 
step_size = 0.25
x_in = -6
x_end = 6


"Forward finite difference"
#initializing variables 
x0 = x_in #initializing input 
y_dot_forw = np.array([]) #initializing forward derivative 
x_forw = np.array([x0]) #initializing forward input

while x0 <= x_end:
    slope = forward_fin_diff(x0, step_size) #(f_k+1 - f_k )/h
    x0 = x0 + step_size # loop the x0 value 
    x_forw = np.append(x_forw, x0) #storing the data
    y_dot_forw = np.append(y_dot_forw,slope ) #storing the data 

plt.plot(x_forw[:-1], y_dot_forw, '-r')  


"Backward finite difference"
#initializing variables 
x0 = x_in
y_dot_back = np.array([])
x_back = np.array([x0])

while x0 <= x_end:
    slope = backward_fin_diff(x0, step_size) #(f_k) -(f_k-1)/h
    x0 = x0 + step_size # loop the x0 value 
    x_back = np.append(x_back, x0) #storing the data
    y_dot_back = np.append(y_dot_back, slope) #storing the data 

plt.plot(x_back[:-1], y_dot_back, '-b')

"Central finite differences"
x0 = x_in
y_dot_central = np.array([])
x_central = np.array([x0])

while x0 <= x_end:
    slope = central_fin_diff(x0, step_size) #(f_k+1 -f_k-1)/2h
    x0 = x0 + step_size
    x_central = np.append(x_central, x0)
    y_dot_central = np.append(y_dot_central, slope)

plt.plot(x_central[:-1], y_dot_central, '-g') #plotting central finite difference 


plt.legend([r'$\dot y$ truth', r'$\dot y$ forward', r'$\dot y$ backward', r'$\dot y$ central']) #specifying curves
plt.title("Plotting finite differences")


    
#%%simple code with no functions
step_size = 0.25

fig2 = plt.figure()
plt.plot(x, fprime, '-k')
plt.grid()
plt.xlabel(r'$x$')
plt.ylabel(r'$\dot y$')
plt.legend([r'$dot y truth'])

"Forward finite difference"
#initializing variables 
x_in = -6
x_end = 6

x0 = x_in
y_dot_forw = np.array([])
x_forw = np.array([x0])

while x0 <= x_end:
    current_value= func(x0) #f_k
    following_value = func(x0 + step_size) #f_k+1
    slope = (following_value - current_value)/step_size #(f_k+1) -(f_k)/h
    x0 = x0 + step_size # loop the x0 value 
    x_forw = np.append(x_forw, x0) #storing the data
    y_dot_forw = np.append(y_dot_forw, slope) #storing the data 

plt.plot(x_forw[:-1], y_dot_forw, '-r')

"Backward finite difference"
#initializing variables 
x0 = x_in
y_dot_back = np.array([])
x_back = np.array([x0])

while x0 <= x_end:
    current_value= func(x0) #f_k
    previous_value = func(x0 - step_size) #f_k+1
    slope = (current_value - previous_value)/step_size #(f_k-1) -(f_k)/h
    x0 = x0 + step_size # loop the x0 value 
    x_back = np.append(x_back, x0) #storing the data
    y_dot_back = np.append(y_dot_back, slope) #storing the data 

plt.plot(x_back[:-1], y_dot_back, '-b')
plt.legend([r'$\dot y$ truth', r'$\dot y$ forward', r'$\dot y$ backward'])
#%% solving ordinary differential equations 
from scipy.integrate import odeint

def ODE_RHS(y,t):
    """ODE right hand side"""
    return -2*y

def Runga_Kutta_1st_order(y0, t0, step_size):
   """approximates y(t0+h) using first-order accurate Euler method
   
   y0 is the initial condition for the 2 dimensional ODE.
   
   t0 is the initial time.
   
   step_size is the discrete time step you'd like to evaluate each numerical approximation at.
   """
   return y0 + ODE_RHS(y0,t0)*step_size

def Runga_Kutta_2nd_order(y0,t0, step_size):
    """ approximates y(t0+h) using second-order accurate Runga Kutta Method
    y0 is the initial condition for the 2 dimensional ODE.
    
    t0 is the initial time.
    
    step_size is the discrete time step you'd like to evaluate each numerical approximation at.
    """
    return y0 + step_size*ODE_RHS(y0 + (step_size/2)*ODE_RHS(y0,t0), t0 + (step_size/2))

def Runga_Kutta_4th_order(y0,t0,step_size):
    """ approximates y(t0+h) using fourth-order accurate Runga Kutta Method
    y0 is the initial condition for the 2 dimensional ODE.
    
    t0 is the initial time.
    
    step_size is the discrete time step you'd like to evaluate each numerical approximation at.
    """
    k1 = ODE_RHS(y0,t0)
    k2 = ODE_RHS(y0 + (step_size/2)*k1, t0 + step_size/2)
    k3 = ODE_RHS(y0 + k2*(step_size)/2, t0 + step_size/2)
    k4 = ODE_RHS(y0 + k3*step_size, t0 + step_size)
    return y0 + (k1+2*k2+2*k3+k4)*step_size/6

"Set-up the problem"
y0 = 3
t0 = 0
tf = 2

"Evaluating exact solution using odeint"
time = np.linspace(t0,tf) 
y_ODE_solver = odeint(ODE_RHS, y0, time) #solution 
#odeint is used like : solution = odeint(ODE, initial condition, time vector)

"Numerical Integration "
 
"First Order Runga Kutta Approximation"
#initializing variables 
current_time = t0
final_time = tf
current_value = y0 #initial condition f[:,0]
step_size = 0.2 
timeline = np.array([t0])
fsol = np.array([y0])

while current_time < final_time-step_size:
    
    #Solve ODE
    #fsol_forward = current_value + ODE_RHS(current_value, current_time)*step_size #forward stepped value of y: y(t0 + h)
    fsol_forward = Runga_Kutta_1st_order(current_value, current_time, step_size) 
    
    #iterate through t and y values
    current_time = current_time + step_size # t = t0 + h, loop through values of time, initializing the next step
    current_value = fsol_forward #loop through values of y, initializing the next step
    
    #append/save solutions
    timeline = np.append(timeline, current_time) #append the timeline to include all values of time which y is evaluated at
    fsol = np.append(fsol, fsol_forward) #append the solution vector to have all values of y

"Second Order Runga Kutta Approximation"
#initializing variables
timeline1 = np.array([t0])
fsol1 = np.array([y0])  
current_time = t0
final_time = tf
step_size = 0.2
current_value = y0

while current_time < final_time-step_size:
    
    #Solve ODE
    fsol1_forward = Runga_Kutta_2nd_order(current_value, current_time, step_size) 
    
    #iterate through t and y values
    current_time = current_time + step_size # t = t0 + h, loop through values of time, initializing the next step
    current_value = fsol1_forward #loop through values of y, initializing the next step
    
    #append/save solutions
    timeline1 = np.append(timeline1, current_time) #append the timeline to include all values of time which y is evaluated at
    fsol1 = np.append(fsol1, fsol1_forward) #append the solution vector to have all values of y
    
"4th order Runga Kutta Approximation"
#initializing variables
timeline = np.array([t0])
fsol2 = np.array([y0])  
current_time = t0
final_time = tf
step_size = 0.2
current_value = y0

while current_time < final_time-step_size:
    
    #Solve ODE
    fsol2_forward = Runga_Kutta_4th_order(current_value, current_time, step_size) 
    
    #iterate through t and y values
    current_time = current_time + step_size # t = t0 + h, loop through values of time, initializing the next step
    current_value = fsol2_forward #loop through values of y, initializing the next step
    
    #append/save solutions
    timeline = np.append(timeline, current_time) #append the timeline to include all values of time which y is evaluated at
    fsol2 = np.append(fsol2, fsol2_forward) #append the solution vector to have all values of y

"Plotting approximations of y(t)"
fig3 = plt.figure()
plt.plot(time, y_ODE_solver, 'k-', linewidth = 2) #plotting exact solution
plt.grid()
plt.xlabel('time')
plt.ylabel(r'$y(t)$')
plt.plot(timeline, fsol, 'g-o', linewidth = 2) #plotting firt order Runga Kutta
plt.plot(timeline1,fsol1, 'r-o', linewidth = 2) #plotting 2nd order Runga Kutta
plt.plot(timeline, fsol2, 'b-o', linewidth = 2) #plotting 4th order Runga Kutta
plt.legend(['Truth', 'Runga Kutta 1st order approximation', 'Runga Kutta 2nd order approximation', 'Runga Kutta 4th order approximation']) #adding the legend :-)


#%%
"Evaluating Errors"
error_in_1st_order = abs(y_ODE_solver - fsol)
error_in_2nd_order = abs(y_ODE_solver - fsol1)
error_in_4th_order = abs(y_ODE_solver - fsol2)

fig4 = plt.figure()
plt.plot(timeline,error_in_1st_order)
#%% evaluating nonlinear pendulumL free, damped, controlled  

def nonlinear_pendulum_free(x,time):
    g = 9.81 #m/s
    l = 3 #m
    xdot = np.zeros(2) #x1 and x2, or theta_dot and theta_double_dot
    xdot[0] = x[1]
    xdot[1] = -g/l*np.sin(x[0])
    return xdot

def pendulum_damped(x,time):
    """dynamics of a damped pendulum"""
    g=9.81
    l = 3
    b = 0.3 #damping coefficient
    xdot = np.zeros(2)
    xdot[0] = x[1]
    xdot[1] = -g/l*np.sin(x[0]) -b*x[1] #can be found by Newton's second law
    return xdot
def RK2(ODE,y0,t0, step_size):
    """generalized 2nd order Runga Kutta function"""
    return y0 + step_size*ODE(y0 + (step_size/2)*ODE(y0,t0), t0 + (step_size/2))

def RK4(ODE,y0, t0, step_size):
    """generalized 4th order Runga Kutta function"""
    k1 = ODE(y0,t0)
    k2 = ODE(y0 + (step_size/2)*k1, t0 + step_size/2)
    k3 = ODE(y0 + k2*(step_size)/2, t0 + step_size/2)
    k4 = ODE(y0 + k3*step_size, t0 + step_size)
    return y0 + (k1+2*k2+2*k3+k4)*step_size/6

x0 = np.array([np.pi/3, 0])
t0 = 0
tf = 15
npoints = 1000
time = np.linspace(t0,tf,npoints)
step_size = .2
solution_undamped = odeint(nonlinear_pendulum_free, x0, time)
solution_damped = odeint(pendulum_damped, x0, time)

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(time, solution_undamped[:,0], c = 'orange')
ax1.set_ylabel(r'$\theta$''- undamped pendulum', fontsize = 8)

ax2 = fig.add_subplot(212)
ax2.plot(time, solution_damped[:,0], c = 'blue')
ax2.set_ylabel(r'$\theta$''-damped pendulum', fontsize = 8)
ax2.set_xlabel('Time (s)')

#%%
def Lorenz63(x,time, sigma, rho, beta):
    """This function returns the state space vector of the Lorenz63 system"""
    x_dot = np.zeros(3)
    x_dot[0] = sigma*(x[1]-x[0])
    x_dot[1] = x[0]*(rho - x[2]) - x[1]
    x_dot[2] = x[0]*x[1] - beta*x[2]
    return x_dot

#initializing variables 
x0 = np.array([5,5,5]) #initial condition
t0 = 0
tf = 20
npoints = 1000
time = np.linspace(t0,tf, npoints) #time vector

#constants
sigma = 10
rho = 28
beta = 8/3

#Assignment 1
solution_lorenz63 = odeint(Lorenz63, x0, time, args = (sigma, rho, beta)) #plotting lorenz solution with specific initial condition
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot3D(solution_lorenz63[:,0], solution_lorenz63[:,1], solution_lorenz63[:,2], 'b')

#Assignment 2, for 20 randomly selected, uniformly distributed, initial conditions
x0_vec = np.random.randint(-20,20, 20) #selecting 20 random points
y0_vec = np. random.randint(-30,30, 20)
z0_vec = np.random.randint(0,50, 20)

fig1 = plt.figure()
ax1 = plt.axes(projection = '3d')

#creating a for loop to solve the Lorenz solution and plot for randomly selected initial conditions
for i_c in range(20):
    x0 = np.array([x0_vec[i_c], y0_vec[i_c], z0_vec[i_c]]) #defining x0 vector 
    Lorenz_random = odeint(Lorenz63, x0,time, args = (sigma, rho, beta)) #solving Lorenz solution
    ax1.plot3D(Lorenz_random[:,0], Lorenz_random[:,1], Lorenz_random[:,2]) #plotting for each random initial condition
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')



