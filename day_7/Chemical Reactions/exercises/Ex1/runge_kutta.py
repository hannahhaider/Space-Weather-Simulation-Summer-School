from numpy.linalg import norm
import numpy as np

def explicit_RK_stepper(f,x,t,h,a,b,c):
    """
        Implementation of generic explicit Runge-Kutta update for explicit ODEs
        
        inputs:
            x - current state 
            t - current time
            f - right-hand-side of the (explicit) ODE to be integrated (signature f(x,t))
            h - step size 
            a - coefficients of Runge-Kutta method (organized as list-of-list (or vector-of-vector))
            b - weights of Runge-Kutta method (list/vector)
            c - nodes of Runge-Kutta method (including 0 as first node) (list/vector)

        outputs: 
            x_hat - estimate of the state at time t+h
    """
    s = len(b)
    ks = [f(x,t)]
    x_hat = x + h*b[0]*ks[-1]
    for i in range(s-1):
        
        y = x + h*sum(a[i][j]*ks[j] for j in range(i+1)) #computing left handside of f(_,t); a is a list of lists (2D) not an array
        ks.append(f(y,t+c[i+1]*h)) #appending right handside
        x_hat += b[i+1]*h*ks[-1] #keep looping and adding the h*b*ks terms until i = s, which is why we started with b[0]*ks[-1]            
        #can also do just: x_hat = x + h*sum(b[i]*ks[i] for i in range(s)) outside of the for loop
    return x_hat # please complete this function 
#f = -2*x
from dormand_prince import a,b,c

def integrate(f, x0, tspan, h, step):
    """
        Generic integrator interface

        inputs:
            f     - rhs of ODE to be integrated (signature: dx/dt = f(x,t))
            x0    - initial condition (numpy array)
            tspan - integration horizon (t0, tf) (tuple)
            h     - step size
            step   - integrator with signature: 
                        step(f,x,t,h) returns state at time t+h 
                        - f rhs of ODE to be integrated
                        - x current state
                        - t current time 
                        - h stepsize

        outputs: 
            ts - time points visited during integration (list)
            xs - trajectory of the system (list of numpy arrays)
    """
    t, tf = tspan
    x = x0
    trajectory = [x0]
    ts = [t]
    while t < tf:
        h_eff = min(h, tf-t)
        x = step(f,x,t,h_eff)
        t = min(t+h_eff, tf)
        trajectory.append(x)
        ts.append(t)
    return trajectory, ts