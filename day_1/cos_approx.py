#!/usr/bin/env python
"""Space 477: Python: I

cosine approximation function
"""
__author__ = 'Hannah Haider'
__email__ = 'hzhaider@ucsd.edu'

import argparse 
from math import factorial
from math import pi
import numpy as np 

def parse_args():
    
    """function to parse input arguments"""
    parser = argparse.ArgumentParser(description = \
                                     'This code approximates the cos(x) function using a taylor series expansion for an input x and a number of points npts used to approximate the function. ')
    
    parser.add_argument('input', nargs = 1, \
                        help = 'Need one input variable x, the angle to approximate cosine', \
                            type = float)
    
    parser.add_argument('-npts', \
                        help = 'another scalar, the number of points used to approximate cos(x)', \
                            type = int, default = 10)
    
    args = parser.parse_args()
    
    return args 
    


def cos_approx(x, accuracy=10):
    """This is a function that approximates cos(x) by summing the first n (accuracy) terms of the Taylor series expansion"""
    Taylor_Series_Exp = [(((-1)**n)/(factorial(2*n)))*(x**(2*n)) for n in range(accuracy + 1)]
    
    return sum(Taylor_Series_Exp)



# Will only run if this is run from command line as opposed to imported
if __name__ == '__main__':  # main code block
    
    args = parse_args()
    print(args)
    
    x = args.input[0] 
    print(x)
    
    accuracy = args.npts
    print(accuracy)

    args = parse_args()
    print(args)
    
    x = args.input[0] 
    print(x)
    
    accuracy = args.npts
    print(accuracy)


    def is_close(value, close_to, eta = 1.e-2):
        """Returns True if approximated value is close to the exact value by a value of eta"""
        comparison = value > close_to - eta and value < close_to + eta
        return comparison 
    
    approx =cos_approx(x,accuracy)
    print("cos_approx(x) = ", approx)
    
 #check whether the approximation is good or not   
    if is_close(approx, np.cos(x)) == True:
        print("This is a good approximation")
    else:
        print("This is a bad approximation")
   
    #print("cos(pi) = ", cos_approx(pi))
    #print("cos(2*pi) = ", cos_approx(2*pi))
    #print("more accurate cos(2*pi) = ", cos_approx(2*pi, accuracy=50))
    #assert is_close(cos_approx(x),np.cos(x))

    print("cos(0) = ", cos_approx(0))
    print("cos(pi) = ", cos_approx(pi))
    print("cos(2*pi) = ", cos_approx(2*pi))
    print("more accurate cos(2*pi) = ", cos_approx(2*pi, accuracy=50))
    assert is_close(cos_approx(0),1), "cos(0) is not 1"

    assert is_close(cos_approx(pi),-1), "cos(pi) is not -1"

    assert is_close(cos_approx(pi),-1), "cos(pi) is not -1"
    #assert is_close(cos_approx(x),np.cos(x))
