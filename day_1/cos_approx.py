#!/usr/bin/env python
"""Space 477: Python: I

cosine approximation function
"""
__author__ = 'Hannah Haider'
__email__ = 'hzhaider@ucsd.edu'

from math import factorial
from math import pi


def cos_approx(x, accuracy=10):
    """This is a function that approximates cos(x) by summing the first n (accuracy) terms of the Taylor series expansion"""
    Taylor_Series_Exp = [(((-1)**n)/(factorial(2*n)))*(x**(2*n)) for n in range(accuracy + 1)]
    return sum(Taylor_Series_Exp)



# Will only run if this is run from command line as opposed to imported
if __name__ == '__main__':  # main code block

    def is_close(value, close_to, eta = 1.e-2):
        """Returns True if approximated value is close to the exact value by a value of eta"""
        comparison = value > close_to - eta and value < close_to + eta
        return comparison 
    
    print("cos(0) = ", cos_approx(0))
    print("cos(pi) = ", cos_approx(pi))
    print("cos(2*pi) = ", cos_approx(2*pi))
    print("more accurate cos(2*pi) = ", cos_approx(2*pi, accuracy=50))
    assert is_close(cos_approx(0),1), "cos(0) is not 1"
    assert is_close(cos_approx(pi),-1), "cos(pi) is not -1"