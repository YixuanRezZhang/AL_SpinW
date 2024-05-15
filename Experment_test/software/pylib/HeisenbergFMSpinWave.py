import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numba import njit, float64
import plotformat as pf

## Spin wave dispersion calculator for Gd assuming a collinear ferromagnet ground state

a, b, c = 3.5126, 4.7457, 7.9171
alpha, beta, gamma = 90, 90, 90

avec1 = np.array([a, 0, 0])
avec2 = np.array([0, b, 0])
avec3 = np.array([0, 0, c])


def cartesian(aa,bb,cc, norm=True):
    # converts aa, bb, cc to cartesian coordinates
    vec = aa*avec1 + bb*avec2 + cc*avec3
    if norm:
        return np.around(np.linalg.norm(vec),4)
    else:
        return vec
    
    
rho = [a/2, -b/2, c*(0.62807-0.37193)] #r1-r2
rhoABC = [1/2, -1/2, (0.62807-0.37193)]


dists = []
# A, B, C, 1-1 or 2-1, NNindex, cartesianDistance

for i in range(-4,5):
    for j in range(-4,5):
        for k in range(-3,4):
            if i != 0 or j != 0 or k != 0:
                dists.append([i,j,k,1,0,cartesian(i,j,k)])
            dists.append([i+rhoABC[0], j+rhoABC[1], k+rhoABC[2], 2, 0,
                            cartesian(i+rhoABC[0], j+rhoABC[1], k+rhoABC[2])])


dists = np.array(dists)
# Sort by distance
dists = dists[np.argsort(dists[:,-1])]

alldists = np.unique(np.sort(dists[:,-1]))

for i in range(len(dists)):
    NNindex = np.squeeze(np.where(alldists == dists[i,-1]))
    dists[i,-2] = int(NNindex)



@njit
def scriptJs(qvec, J):
    '''create J1 and J2'''
    
    # Define the q vectors and the energies
    qvrlu = qvec*2*np.pi
    en1 = np.zeros_like(qvec[:,0], dtype=np.complex128)
    en2 = np.zeros_like(qvec[:,0], dtype=np.complex128)
    
    # Loop through all neighbors
    for i in range(len(dists)):
        d = dists[i]
        if d[-2] >= len(J):  #we got to highest J defined
            break
        
        rmr = d[:3] #distance between sites
        Ji = J[int(d[-2])]  #exchange constant
        
        ### Removed factor of 2 compared to Gd.
        if d[-3] == 1: #It's a site-1 to site-1 exchange
            en1 += Ji*np.exp(-1j*np.dot(qvrlu,rmr))
        elif d[-3] == 2: #It's a site-1 to site-2 exchange
            en2 += Ji*np.exp(-1j*np.dot(qvrlu,rmr))
        
    return en1, en2

qzero = np.array([[0,0,0]], dtype=np.float64)
Jz = 3/2
Anisotropy = 0.0

@njit
def hw_ac(qvec, J):
    '''accoustic magnon branch. J is a list of exchange values 
    from nearest neighbor to further neighbor'''
    j10, j20 = scriptJs(qzero, J)
    j1, j2 = scriptJs(qvec, J)
    return -Jz*(j10 + j20 - j1 - np.abs(j2)) + Anisotropy
    
@njit
def hw_op(qvec, J):
    '''optical magnon branch. J is a list of exchange values 
    from nearest neighbor to further neighbor'''
    j10, j20 = scriptJs(qzero, J)
    j1, j2 = scriptJs(qvec, J)
    return -Jz*(j10 + j20 - j1 + np.abs(j2)) + Anisotropy



################################################### Gd specific stuff
import warnings
### Import stuff for fit

KDfile = '/home/1o1/Documents/Projects_Columbia/CrSBr/SEQUOIA/CSBModes_ExtractedFromSEQ.txt'
MyData = np.genfromtxt(KDfile, skip_header=1, delimiter='\t', unpack=True)

E1i = 7
dE1i = 8
E2i = 13
dE2i = 14

# eliminate zeros
MyData[E2i][MyData[E2i] == 0] *= np.nan


## Find where the discontinuities are

# SplitData = [0]
# for i,md in enumerate(MyData[:,1:].T):
#     if np.linalg.norm(md[:3] - MyData[:3,i]) > 0.5:
#         SplitData.append(i)
# SplitData.append(len(MyData[0]))


####################################################
@njit
def calculateDispersionFromFile(data, JJ):
    ac = hw_ac(data[:3].T, JJ)
    op = hw_op(data[:3].T, JJ)
    return ac, op

#@njit
def globalerror(dat, JJ):
    llnn = np.count_nonzero(~np.isnan(dat[E1i])) +\
            np.count_nonzero(~np.isnan(dat[E2i])) - np.count_nonzero(JJ)
    acc, opt = calculateDispersionFromFile(dat, JJ)
    
    err11 = (np.real(opt) - dat[E1i])**2/(dat[dE1i]**2)
    err22 = (np.real(acc) - dat[E2i])**2/(dat[dE2i]**2)
    err12 = (np.real(acc) - dat[E1i])**2/(dat[dE1i]**2)
    err21 = (np.real(opt) - dat[E2i])**2/(dat[dE2i]**2)
    
    err1 = np.nanmin(np.array([err11, err12]), axis=0)
    err2 = np.nanmin(np.array([err22, err21]), axis=0)
    
    return (np.nansum(err1) + np.nansum(err2))/llnn
    #return 1/(1/np.nansum(err1) + 1/np.nansum(err2))

def chisquared(JJ):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return globalerror(MyData, JJ)


def plotresult(dat, JJ):
    ac1, op1 = calculateDispersionFromFile(dat, JJ)
    plt.figure(figsize=(5.4,3.4))
    plt.plot(ac1, color=pf.cpal2[0])
    plt.plot(op1, color=pf.cpal2[1])
    plt.plot(dat[E2i], marker='.', ls='none', color=pf.cpal2[0])
    plt.plot(dat[E1i], marker='.', ls='none', color=pf.cpal2[1])
    