from sage.stats.distributions.discrete_gaussian_integer import         DiscreteGaussianDistributionIntegerSampler as DGaussZ

# import Python packages
from mpmath import sqrt, jtheta, exp
from math import comb, pi, floor
import numpy as np
import random 
import time

def nome(s):
    return exp(-1*pi/s**2)

def scaled_psi_4(s):
    q=nome(s)**4
    return (1/2)*jtheta(2, 0, q)

def theta_BW16(s):
    q = nome(s)
    theta3 = jtheta(3,0,q**4)
    theta2 = jtheta(2,0,q**4)
    theta4 = jtheta(4,0,q**4)
    theta_series = (1/2)*theta3**16+15*(theta3**8)*(theta2**8)+(1/2)*theta2**16+(1/2)*theta4**16
    return theta_series

def coset_prob_BW16(s):
    q = nome(s)
    theta3 = jtheta(3,0,q**4)
    theta2 = jtheta(2,0,q**4)
    theta4 = jtheta(4,0,q**4)
    theta_series = theta_BW16(s)
    table = []
    coset_0 = ((1/2)*theta3**16+(1/2)*theta4**16)/theta_series
    coset_8 = (15*(theta2**8)*theta3**8)/theta_series
    coset_16 = ((1/2)*theta2**16)/theta_series
    table.extend([coset_0, coset_8, coset_16])
    return table

def theta_dn(n,s):
    q = nome(s)
    theta3 = jtheta(3,0,q**4)
    theta2 = jtheta(2,0,q**4)
    theta_series = mpf('0')
    for m in range(1,n/2+1):
        theta_series += comb(n,2*m)*(theta3**(n-2*m))*(theta2**(2*m))
    return theta_series

def coset_prob_dn(n, s):
    table=[]
    q = nome(s)
    theta3 = jtheta(3,0,q**4)
    theta2 = jtheta(2,0,q**4)
    for m in range(1,n/2+1):
        p_2m = comb(n,2*m)*(theta3**(n-2*m))*(theta2**(2*m))/theta_dn(n, s)
        table.append(p_2m)
    return table

def weight_enum_e(n, s):
    sum = 0
    theta3 = jtheta(3,0,nome(s)**4)
    theta2 = jtheta(2,0,nome(s)**4)
    for i in range(floor((n-1)/2)):
        sum += comb(n-1, 2*i)*(theta3**(n-1-2*i))*(theta2**(2*i))
    return sum

def weight_enum_o(n, s):
    sum = 0
    theta3 = jtheta(3,0,nome(s)**4)
    theta2 = jtheta(2,0,nome(s)**4)
    for i in range(1,floor((n+1)/2)):
        sum += comb(n-1, 2*i-1)*(theta3**(n-2*i))*(theta2**(2*i-1))
    return sum

def sample_dn(n, s):
    indices = [i for i in range(n)]
    m = random.choices(indices[1:n/2+1], weights=coset_prob_dn(n,s))
    subset = random.sample(indices, k=2*m[0])
    x = [None] * n
    for j in range(len(subset)):
        x[subset[j]] = 2 * DGaussZ(sigma=s/(2*sqrt(2*pi)),c=1/2)()
    not_subset = list(set(indices)-set(subset))
    for j in range(len(not_subset)):
        x[not_subset[j]] = 2 * DGaussZ(sigma=s/(2*sqrt(2*pi)))()
    return x

def sample_dn_special_shift(n, s, t):
    c = list(random.sample(list(codes.ParityCheckCode(GF(2), n-1)), k=1)[0])
    c_real = [Integer(i) for i in c]
    x = [None] * n
    for i in range(n):
        x[i] = 2 * DGaussZ(sigma=s/(2*sqrt(2*pi)),c=(c_real[i]+t[i])/2)() # scaling issue
    return x

def sample_dn_bar(n, s):
    indices = [i for i in range(n)]
    m = random.choices(indices[1:n/2+1], weights=coset_prob_dn(n,s))[0] # weight_ls is same as in normal Dn alg
    q = nome(s)
    if m < n/2:
        num_heads = (jtheta(2,0,q**4)**2)
        p_heads = num_heads/(num_heads+(2*m)/(n-2*m)*(jtheta(3,0,q**4)**2))
    else:
        p_heads = 0

    coin = random.choices(['heads', 'tails'], weights=[p_heads,1-p_heads])[0]

    x = [None] * n
    if coin == 'heads':
        subset = random.sample(indices[1:], k=2*m)+[0]
        for j in range(len(subset)):
            x[subset[j]] = 2 * DGaussZ(sigma=s/(2*sqrt(2*pi)),c=1/2)()
        not_subset = list(set(indices)-set(subset))
        for j in range(len(not_subset)):
            x[not_subset[j]] = 2 * DGaussZ(sigma=s/(2*sqrt(2*pi)))()
    else:
        subset = random.sample(indices[1:], k=2*m-1)
        for j in range(len(subset)):
            x[subset[j]] = 2 * DGaussZ(sigma=s/(2*sqrt(2*pi)),c=1/2)()
        not_subset = list(set(indices)-set(subset))
        for j in range(len(not_subset)):
            x[not_subset[j]] = 2 * DGaussZ(sigma=s/(2*sqrt(2*pi)))()
    return x

'''
Precompute table of codewords of weight 8 in Reed-Muller code
'''

C1 = codes.BinaryReedMullerCode(1,4)

weight_8 = []
for word in C1:
    if list(word).count(1) == 8:
        weight_8.append(list(word))

def sample_BW16(s, weights_prob):
    w = random.choices([0,8,16], weights=weights_prob)[0]
    if w == 0: 
        samp = 2*np.array(sample_dn(16, s/4))
        return samp
    elif w == 16:
        samp = 2*np.array(sample_dn_special_shift(16, s/4, t=[1/2]*16))
        return samp
    else: 
        c = random.sample(weight_8, k=1)[0]

        # throw a biased coin with prob even, sample over even-even part, otherwise odd-odd part
        num_even = (2**7)*(scaled_psi_4(s)**8)*theta_dn(8, s)
        denom_even = num_even + (2**7)*(scaled_psi_4(s)**8)*(jtheta(2, 0, nome(s)**4)*weight_enum_e(8,s)+jtheta(3, 0, nome(s)**4)*weight_enum_o(8,s))
        p_even = num_even/denom_even

        decomp = random.choices(['even-even', 'odd-odd'], weights=[p_even,1-p_even])
        supp_J = [i for i in range(16) if c[i]==1] # set supports 
        not_supp = list(set([i for i in range(16)])-set(supp_J))

        if decomp == 'even-even': # sample from even-even part
            t1 = [1/2] * 8
            x = sample_dn_special_shift(8, s/4, t1) # double check if we need to scale this by 2?
        
            y = 2*np.array(sample_dn(8, s/4))
        else: 
            t1 = [3/2] + [1/2] * 7
            x = sample_dn_special_shift(8, s/4, t1)
        
            y = 2*np.array(sample_dn_bar(8, s/4))

        samp = [None] * 16
        for j in supp_J:
            samp[j] = x[supp_J.index(j)]
        for j in not_supp:
            samp[j] = y[not_supp.index(j)]

    return samp

coset_prob = coset_prob_BW16(2.3)

start_time1 = time.perf_counter ()
for i in range(100000):
    sample_BW16(2.3, coset_prob)
end_time1 = time.perf_counter ()
time1 = end_time1 - start_time1

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

def espitau_E8_shift(s,s_prime,shift): 
    '''
    s is the smoothing parameter of D8, s_prime is the smoothing parameter of E8
    '''
    b = random.sample([0,1], k=1)[0]
    bh = np.array([b*1/2 for i in range(8)])
    x = espitau_dn_shift(8, s,shift-bh)
    samp = bh+x
    prob = rho_vec(samp, s_prime)/rho_vec(samp, s)
    resp = random.choices(['accept', 'reject'], weights=[prob, 1-prob])[0]
    if resp == 'accept':
        return samp
    else:
        samp = espitau_E8_shift(s,s_prime,shift)
        return samp

R = np.array([[1,1,0,0,0,0,0,0],
              [1,-1,0,0,0,0,0,0],
              [0,0,1,1,0,0,0,0],
              [0,0,1,-1,0,0,0,0],
              [0,0,0,0,1,1,0,0],
              [0,0,0,0,1,-1,0,0],
              [0,0,0,0,0,0,1,1],
              [0,0,0,0,0,0,1,-1]])

B = np.array([[2, 0, 0, 0, 0, 0, 0, 0],
              [0, 2, 0, 0, 0, 0, 0, 0],
              [0, 0, 2, 0, 0, 0, 0, 0],
              [0, 0, 0, 2, 0, 0, 0, 0],
              [1, 1, 1, 0, 1, 0, 0, 0],
              [1, 0, 1, 1, 0, 1, 0, 0],
              [1, 0, 0, 1, 1, 0, 1, 0],
              [1, 0, 0, 0, 1, 1, 0, 1]])

RB=np.dot(B,R)
RB.echelon_form()

# compute the coset representatives for BW8/RBW8
A = []

for a1, a3, a4, a5, a6, a7 in product(range(2), range(2), range(2), range(4), range(2), range(4)):
    rep = a1*B[1,:]+a3*B[3,:]+a4*B[4,:]+a5*B[5,:]+a6*B[6,:]+a7*B[7,:]
    A.append(rep)

len(A)

def espitau_BW16(s):
    alpha = np.array(random.sample(A, k=1)[0])
    m = np.append(alpha, alpha)
    u1 = np.dot(espitau_E8_shift(2.97, s, -m[0:8]), R) # sampling BW8 and then rotate to sample RBW8
    u2 = np.dot(espitau_E8_shift(2.97, s, -m[8:16]),R) 
    u = np.append(u1, u2)
    samp = [u[i] + m[i] for i in range(16)]
    return samp

start_time2 = time.perf_counter ()
for i in range(100000):
    espitau_BW16(2.85)
end_time2 = time.perf_counter ()
time2 = end_time2 - start_time2

print("Our sampler is", time2/time1, "times as fast as the sampler by Espitau et. al.")

'''
Note: in order to obtain a valid probability prob = rho_vec(samp, s_prime)/rho_vec(samp, 2.97) we have that s_prime has to be at most 2.97.
The smoothing parameter of BW16 is about 2.3 for epsilon = 2^-36. However, when we tried to run the algorithm with 2.3 as input, we obtained error 
messages about reaching maximum depth recursion. We have to choose a sampling width above the smoothing of BW16, but smaller than 2.97 so we have a 
small available range.
'''