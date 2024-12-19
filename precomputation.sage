# import Python packages
from mpmath import sqrt, jtheta, exp
from math import comb, pi

def nome(s):
'''
Compute the nome q = exp(-1*pi/s**2)

:param s: sampling width

:returns: a real value
'''
    return exp(-1*pi/s**2)

def theta_dn(n,s):
'''
Compute the theta series of Dn

:param n: lattice dimension
:param s: sampling width

:returns: a real value
'''
    q = nome(s)
    theta3 = jtheta(3,0,q**4)
    theta2 = jtheta(2,0,q**4)
    theta_series = mpf('0')
    for m in range(1,n/2+1):
        theta_series += comb(n,2*m)*(theta3**(n-2*m))*(theta2**(2*m))
    return theta_series

def coset_prob_dn(n, s):
'''
Compute the probability that a vector lies in a 
coset of Dn represented by a codeword of weight w=1,...,n/2

:param n: lattice dimension
:param s: sampling width

:returns: a list of real values
'''
    table=[]
    q = nome(s)
    theta3 = jtheta(3,0,q**4)
    theta2 = jtheta(2,0,q**4)
    for m in range(1,n/2+1):
        p_2m = comb(n,2*m)*(theta3**(n-2*m))*(theta2**(2*m))/theta_dn(n, s)
        table.append(p_2m)
    return table