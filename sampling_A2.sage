from sage.stats.distributions.discrete_gaussian_integer import         DiscreteGaussianDistributionIntegerSampler as DGaussZ

# import Python packages
from mpmath import sqrt, jtheta, exp
from math import pi
import numpy as np
import time

def nome(s):
    return exp(-1*pi/s**2)

def coset_prob_A2(s):
    q = nome(s)
    theta_A2 = jtheta(3,0,q)*jtheta(3,0,q**3)+jtheta(2,0,q)*jtheta(2,0,q**3)
    prob = jtheta(3,0,q)*jtheta(3,0,q**3)/theta_A2
    return [prob, 1-prob]

def sample_A2(s, weights_prob):
    cosets =[[0,0], [1/2, sqrt(3)/2]]
    t = random.choices(cosets, weights=weights_prob)
    if t == [0,0]:
        x = np.array(DGaussZ(sigma=s/sqrt(2*pi))())
        np.append(x, sqrt(3)*DGaussZ(sigma=s/sqrt(3*2*pi))())
    else:
        x = np.array(DGaussZ(s/sqrt(2*pi),c=1/2)())
        np.append(x, sqrt(3)*DGaussZ(sigma=s/sqrt(3*2*pi), c=1/2)())
    return x

start_time = time.perf_counter ()
for i in range(400000):
    sample_A2(1.46)
end_time = time.perf_counter ()
print(end_time - start_time, "seconds")