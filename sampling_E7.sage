from sage.stats.distributions.discrete_gaussian_integer import         DiscreteGaussianDistributionIntegerSampler as DGaussZ

# import Python packages
from mpmath import sqrt, jtheta, exp
from math import pi
import numpy as np
import random 
import time

def nome(s):
    return exp(-1*pi/s**2)
    
def theta_E7(s):
    q = nome(s)
    theta3 = jtheta(3,0,q**4)
    theta2 = jtheta(2,0,q**4)
    theta_E7 = theta3**7+7*(theta3**4)*(theta2**3)+7*(theta3**3)*(theta2**4)+theta2**7
    return theta_E7

def coset_prob_E7(s):
    table=[]
    q = nome(s)
    theta3 = jtheta(3,0,q**4)
    theta2 = jtheta(2,0,q**4)
    theta_series = theta_E7(s)
    coset_0 = theta3**7/theta_series
    coset_3 = 7*(theta3**4)*(theta2**3)/theta_series
    coset_4 = 7*(theta3**3)*(theta2**4)/theta_series
    coset_7 = theta2**7/theta_series
    table.extend([coset_0, coset_3, coset_4, coset_7])
    return table

C = codes.HammingCode(GF(2),3)
weight_3 = [[1, 0, 0, 0, 0, 1, 1], [0, 1, 0, 0, 1, 0, 1], [0, 0, 1, 0, 1, 1, 0],
            [1, 1, 1, 0, 0, 0, 0], [1, 0, 0, 1, 1, 0, 0], [0, 1, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 0, 1]]

weight_4 = [[1, 1, 0, 0, 1, 1, 0], [1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1], [1, 1, 0, 1, 0, 0, 1], [1, 0, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 0, 0]]

def sample_E7(s, weights_prob):
    w = random.choices([0,3,4,7], weights=weights_prob)
    if w[0] == 3:
        c = random.sample(weight_3, k=1)
        x = np.array([2*DGaussZ(sigma=s/(2*sqrt(2*pi)), c=c[0][i]/2)() for i in range(7)])
    elif w[0] == 4:
        c = random.sample(weight_4, k=1)
        x = np.array([2*DGaussZ(sigma=s/(2*sqrt(2*pi)), c=c[0][i]/2)() for i in range(7)])
    elif w[0] == 0: 
        x = np.array([2*DGaussZ(sigma=s/(2*sqrt(2*pi)))() for i in range(7)])
    elif w[0] == 7:
        x = np.array([2*DGaussZ(sigma=s/(2*sqrt(2*pi)),c=1/2)() for i in range(7)])
    return x

start_time1 = time.perf_counter ()
for i in range(100000):
    sample_E7(1.75, coset_prob_E7(1.75))
end_time1 = time.perf_counter ()
print(end_time1 - start_time1, "seconds")