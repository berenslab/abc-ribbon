import numpy as np
import scipy as scp
import scipy
from scipy.stats import rv_discrete
import scipy.signal 

from random import *
from math import *

import pandas as pd


# define beta binomial distribution

class beta_binomial(rv_discrete):
    """
    creating betabinomial distribution by defining its pmf
    """
    def _pmf(self, k, a, b, n):
        return scp.special.binom(n,k)*scp.special.beta(k+a,n-k+b)/scp.special.beta(a,b)
    
betabinomial = beta_binomial(name="betabinomial")


def sample_Beta_Binomial(a, b, n, size=None):
    """
    sample from beta binomial using compound distribution property
    # faster than using explicitly the pmf and involved beta functions
    """
    p = np.random.beta(a, b, size=size)
    r = np.random.binomial(n, p)
    return r


# resampling of the stimulus 

def resampleCon(stim, dtstim):
    """
    resamples the stimulus into the specific time resolution of 10ms(!) of the simulation
    ---
    :param stimWave: original stim
    :param dtstim: time resolution of stim
    ---
    :return: resampled values
    """

    binsize = round(0.01 / (dtstim))  # 0.01 as aim resolution
    stimDF = pd.DataFrame({"stim": stim})
    rollMean = stimDF.rolling(window=binsize, center=True).mean()  # same len as initial stimDF
    reSampInds = np.arange(0, len(stim), binsize)  # actual resampling indeces 10 or 100 or...
    lenRe = len(reSampInds)
    resampVals = np.zeros(lenRe)
    asArray = rollMean.values
    resampVals[0] = asArray[int(reSampInds[1])]  # hack to avoid nan in rolling average
    for i in range(0, lenRe):
        resampVals[i] = asArray[int(reSampInds[i])]
    resampVals[0] = resampVals[1] # to avoid nans
    return resampVals


"""
The core model
"""

def sigmoid(x,k,x0):
    """
    define a sigmoid that converts calcium concentration to release probability
    (for ease I set this to range mostly from 0-1, with min a 50)
    ---
    :param x: input value
    :param k: slope of the sigmoid
    :param x0: half activation point: sigmoid(x_half) = 0.5
    ---
    :return: y
    """
    y = (1/(1+np.exp(-k*(x-x0)))) * 0.999
    return y + 0.001

def sampleRelease(docked,releaseP):
    """
    sample number of vesicles released from dock
    !! For non-correlated vesicles!!!
    ---
    :param docked: amount of docked vesicles
    :param releaseP: relea
    ---
    :return: int: choice: released vesicles
    """
    docked = np.int(docked)
    choice = np.random.binomial(docked,releaseP)
    return choice


def sampleReleaseCorrelated(docked,releaseP,rho):
    """
    sample number of vesicles released from dock, correlated version BetaBinomial
    rho is the correlation, between 0 and 1 (almost 1/0 values can produce errors)
    holding ro a fixed value, meaning constant correlation between vesicles
    ---
    :param docked: number of docked vesicles
    :param releaseP: relea
    ---
    :return: int: choice: released vesicles
    """
    if releaseP < 10e-6 or docked ==0:
        return 0
    else:
        #p = releaseP * (docked/15) # possible adaption st. small events occure less often 
        p = np.min((releaseP, 0.999)) # avoids numerical instability for releseP almost 1
        rho_norm = max(0.001, min(rho, 0.999))
        c = 1/rho_norm - 1
        a = p*c
        b = c-a
        #choice = betabinomial.rvs(a,b,docked)
        choice = sample_Beta_Binomial(a,b,docked)
        return choice

def sampleMovementToDock(ribboned,dockP):
    """
    sample number of vesicles moved from ribbon to dock
    ---
    :param ribboned: number of ribboned vesicles
    :param dockP: probability for movement
    ---
    :return: int: choice: moved vesicles
    """
    ribboned = np.int(ribboned)
    choice = np.random.binomial(ribboned,dockP)
    return choice

def sampleMovementToRibbon(ribbonLambda):
    """
    sample vesicles moved from cytoplasm to dock
    by a Poisson distribution
    ---
    :param ribbonLambda: param for poisson distribution
    ---
    :return: int: moved vesicles
    """
    choice = np.random.poisson(ribbonLambda)
    return choice

# actual simulation
def runOne( params, stim, correlated=True):
    """
    run one simulation of the ribbon, assuming dt=10ms
    ---
    params: parameters for the ribbon
    s : stimulus (as Ca) according to t
    ---
    correlated: if True: use BetaBinomial distribution to sample release, else: Binomial
    ---
    returns: release of ribbon (according to rescaled t)
    """
    #filtT = np.arange(0.01,300, dtstim) # in ms
    #filt = cosFilter(filtT,a,c,phi,w*wsign, lagms)
    k = params[0]
    x0 = params[1]
    dockP = params[2]
    ribbonLambda = params[3]
    dockMax = params[4]
    ribbonMax = params[5]
    rho = params[6]
    #stimCon = scipy.signal.convolve(s,filt,mode="full")[:len(s)]/(1/dtstim *10)
    #stimVals = resampleCon(stimCon,t, dtstim)
    releaseP = sigmoid(stim, k, x0)
    releaseP[0] = 0
    nSteps = len(releaseP)
    released = np.zeros(nSteps)  # [[] for i in range(0,nSteps)]
    docked = np.zeros(nSteps)  # [[] for i in range(0,nSteps)]
    ribboned = np.zeros(nSteps)  # [[] for i in range(0,nSteps)]
    released[0] = 0
    docked[0] = dockMax
    ribboned[0] = ribbonMax
    if correlated == True: 
        for i in range(0, nSteps - 2):

            #releaseChoice = sampleRelease(docked[i], releaseP[i])  # sample release
            releaseChoice = sampleReleaseCorrelated(docked[i], releaseP[i], rho)  # sample release correlated


            released[i + 1] = releaseChoice  # update released
            docked[i + 1] = docked[i] - releaseChoice  # update docked

            dockChoice = sampleMovementToDock(ribboned[i], dockP)  # sample movement from ribbon to dock
            if (dockChoice + docked[i + 1] > dockMax):
                dockChoice = dockMax - docked[i + 1]  # can't go beyon max filled
            docked[i + 1] = dockChoice + docked[i + 1]  # update docked
            ribboned[i + 1] = ribboned[i] - dockChoice  # update ribboned

            # select b
            ribbonChoice = sampleMovementToRibbon(ribbonLambda)  # sample movement to ribbon
            if (ribbonChoice + ribboned[i + 1] > ribbonMax):
                ribbonChoice = ribbonMax - ribboned[i + 1]  # can't overfill the ribbon
            ribboned[i + 1] = ribbonChoice + ribboned[i + 1]  # update ribbon
    else:
        for i in range(0, nSteps - 2):

            releaseChoice = sampleRelease(docked[i], releaseP[i])  # sample release
            #releaseChoice = sampleReleaseCorrelated(docked[i], releaseP[i])  # sample release correlated


            released[i + 1] = releaseChoice  # update released
            docked[i + 1] = docked[i] - releaseChoice  # update docked

            dockChoice = sampleMovementToDock(ribboned[i], dockP)  # sample movement from ribbon to dock
            if (dockChoice + docked[i + 1] > dockMax):
                dockChoice = dockMax - docked[i + 1]  # can't go beyon max filled
            docked[i + 1] = dockChoice + docked[i + 1]  # update docked
            ribboned[i + 1] = ribboned[i] - dockChoice  # update ribboned

            # select b
            ribbonChoice = sampleMovementToRibbon(ribbonLambda)  # sample movement to ribbon
            if (ribbonChoice + ribboned[i + 1] > ribbonMax):
                ribbonChoice = ribbonMax - ribboned[i + 1]  # can't overfill the ribbon
            ribboned[i + 1] = ribbonChoice + ribboned[i + 1]  # update ribbon

    return released # ,docked,ribboned



# linear filter part (cone kernel)
#assume:
#taus are from PR kernel (Baden et al.,2014) but time stretching


def cone_kernel_scale(scale):
    """
    PR biphasic kernel
    :param scale: time scale 
    :return: normalized kernel with dt=1ms
    """
    # all variables in sec
    t = np.arange(0,0.3,0.001)

    tau_r = 0.05 * scale # fast:.04# orig: 0.07 # good 0.05
    tau_d = 0.05 *scale # fast:.035#orig: 0.07 # 0.05
    phi = -np.pi * (0.2/1.4) * scale # /5 * scale #10 #orig:-np.pi/5# good: 7
    tau_phase = 100 # typo in the Baden paper ? (should it be 100?)
    kernel = -(t/tau_r)**3 / (1+t/tau_r) * np.exp(-(t/tau_d)**2) * np.cos(2*np.pi*t / phi + tau_phase)
    return kernel/ np.max(np.abs(kernel)) #  / np.abs(np.sum(kernel))





def runOne_with_kernel(params_all, light, celltype=-1):
    """
    runs the model runOne with light preprocessing
    ---
    celltype -1 indicating an OFF cell
    ATTENTION: assumes dt(light) = 1ms, but returns response with dt=10ms
    """
    params = params_all[:-1]
    kernel_scale = params_all[-1]
    # compute stimulus
    # convolve to get Ca trace
    light_ca_kernel =  - celltype * cone_kernel_scale(kernel_scale)
    stimCon = scp.signal.convolve(light, light_ca_kernel, mode="full")[:len(light)]  # /(1/dtstim *10)
    # resample and normalize
    dtstim = 0.001 # in sec
    Ca_raw = resampleCon(stimCon, dtstim)
    # normalize stim
    stim = (Ca_raw - np.min(Ca_raw)) / np.max(np.abs(Ca_raw - np.min(Ca_raw)))

    # run the simulation
    rel1 = runOne(params, stim, correlated=True)
    return rel1