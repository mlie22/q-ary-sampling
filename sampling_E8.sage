from sage.stats.distributions.discrete_gaussian_integer import         DiscreteGaussianDistributionIntegerSampler as DGaussZ

# import Python packages
from mpmath import sqrt, jtheta, exp
from math import pi
import numpy as np
import random 
import time

attach('simulations_dn.sage')

def theta_E8(s):
    q = nome(s)
    theta3 = jtheta(3,0,q**4)
    theta2 = jtheta(2,0,q**4)
    theta_E8 = theta3**8+14*(theta3**4)*(theta2**4)+theta2**8
    return theta_E8

def coset_prob_E8(s):
    table=[]
    q = nome(s)
    theta3 = jtheta(3,0,q**4)
    theta2 = jtheta(2,0,q**4)
    theta_series = theta_E8(s)
    coset_0 = theta3**8/theta_series
    coset_4 = 14*(theta3**4)*(theta2**4)/theta_series
    coset_8 = theta2**8/theta_series
    table.extend([coset_0, coset_4, coset_8])
    return table

C = codes.BinaryReedMullerCode(1,3)

weight_4 = [[0, 1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 1, 0], [0, 0, 1, 1, 0, 0, 1, 1], [1, 1, 0, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 1, 0, 0, 1], [0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 1, 0, 1, 0], [1, 0, 1, 0, 0, 1, 0, 1], [0, 0, 1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0, 1, 1], 
            [0, 1, 1, 0, 1, 0, 0, 1], [1, 0, 0, 1, 0, 1, 1, 0]]

def sample_E8(s, weights_prob):
    '''
    The probabilities of selecting a weight w should be precomputed using the coset_prob_E8 function
    '''
    w = random.choices([0,4,8], weights=weights_prob)
    if w[0] == 4:
        c = random.sample(weight_4, k=1)
        x = np.array([2*DGaussZ(sigma=s/(2*sqrt(2*pi)), c=c[0][i]/2)() for i in range(8)])
    elif w[0] == 0: 
        x = np.array([2*DGaussZ(sigma=s/(2*sqrt(2*pi)))() for i in range(8)])
    elif w[0] == 8:
        x = np.array([2*DGaussZ(sigma=s/(2*sqrt(2*pi)),c=1/2)() for i in range(8)])
    return x

def rho_vec(v, s):
    v_t = np.transpose(v)
    Sigma = (1/s**2)*np.identity(len(v))
    prod = np.dot(Sigma,v)
    return exp(-1*pi*np.dot(v_t,prod))

def espitau_dn_shift(n, s, t):
    x = np.empty(n)
    x[0] += 1
    while (sum(x) % 2) != 0:
        x = np.array([DGaussZ(sigma=s/(sqrt(2*pi)), c=t[i])() for i in range(n)])
    return x

def espitau_E8(s, s_prime):
    ''' 
    s is the smoothing parameter of D8, s_prime is the smoothing parameter of E8
    '''
    b = random.sample([0,1], k=1)[0]
    bh = np.array([b*1/2 for i in range(8)])
    x = espitau_dn_shift(8, s,-bh)
    samp = bh+x
    prob = rho_vec(samp, s_prime)/rho_vec(samp, s)
    resp = random.choices(['accept', 'reject'], weights=[prob, 1-prob])[0]
    if resp == 'accept':
        return samp
    else:
        samp = espitau_E8(s_prime, s)
        return samp

start_time1 = time.perf_counter ()
for i in range(100000):
    sample_E8(2.2, coset_prob_E8(2,2))
end_time1 = time.perf_counter ()
print(end_time1 - start_time1, "seconds")

time1 = end_time1 - start_time1

start_time2 = time.perf_counter ()
for i in range(100000):
    espitau_E8(2.97, 2.2)
end_time2 = time.perf_counter ()
print(end_time2 - start_time2, "seconds")

time2 = end_time2 - start_time2

print("Our sampler is", time2/time1, "times as fast as the sampler by Espitau et. al.")

'''
This is due to the fact that the probability of rejection is sometimes (much) lower than 1/11, 
so we may sometimes require more repetitions to accept. See the below example:
'''

b = random.sample([0,1], k=1)[0]
bh = np.array([b*1/2 for i in range(8)])
x = espitau_dn_shift(8, 2.97, -bh)
samp = bh+x
prob = rho_vec(samp, 2.2)/rho_vec(samp, 2.97)
print(prob)