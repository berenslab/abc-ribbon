# -*- coding: utf-8 -*-

from ribbonv2 import *
import multiprocessing
from scipy.stats import *
import numpy as np
import scipy as scp
import warnings


# New stuff
# W corresponds to filter fitting, R to ribbon Fitting


"""
assume:
taus are from PR kernel (Baden et al.,2014) but time stretching
"""

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




def runManyRibbon(light, g,gData,dataSS, sampsParams, batchsize, w, stim_kind='light', celltype=-1):
    """
    run simulations with sampled ribbon params and constant filter weights
    and calculate the loss fct based on the "true" data: gData and dataSS
    ---
    :param stim: input stimulus ('light'), dt = 1ms
    :param g: (gaussian) kernel, used for trace * g for SS
    :param gData: convolved Data [n, tpts]
    :param dataSS: summary statistics of data
    :param t: time
    :param light_ca_kernel: kernel to convolve light signal with, dt = 1ms
    :param params: sampled parameters for the ribbon, includint rho and kernel scale (as last param)
    :param batchsize: how many simulations per parameter set to run
    :param w: weights for summary statistics
    ---
    :return: samples in ordered form with distance to dataSS
    ---
    !!! this functions takes different dt:
        for stim: dt = 1ms (and upsample this before running model)
        for data: dt = 10ms (like output of model)
    """



    """
    prepare stimulus 
    # rescale and convolve light stimulus with kernel
    """

    if stim_kind=='ca':
        stim = light




    datadim = len(np.shape(gData)) # 1 or 2 depending if only one, or multiple "true" traces
    Nsamps =  len(sampsParams[0,:]) #number of samples
    dSamps = np.zeros(Nsamps) # array for distances
    for i in range(0,Nsamps):
        params_all = sampsParams[:,i]
        params = params_all[:-1]
        kernel_scale = params_all[-1]

        # compute stimulus
        if stim_kind=='light':
            # convolve to get Ca trace
            light_ca_kernel =  - celltype * cone_kernel_scale(kernel_scale)
            stimCon = scp.signal.convolve(light, light_ca_kernel, mode="full")[:len(light)]  # /(1/dtstim *10)
            # resample and normalize
            dtstim = 0.001 # in ms
            Ca_raw = resampleCon(stimCon, dtstim)
            # normalize stim
            stim = (Ca_raw - np.min(Ca_raw)) / np.max(np.abs(Ca_raw - np.min(Ca_raw)))

        # run number of simulations according to the batchsize
        rel1 = runOne( params, stim, correlated=True)
        rel = np.zeros((batchsize,len(rel1) ))
        rel[0] = rel1
        for j in range(1,batchsize):
            #rel[j] = runOneNew(t,c,a,phi,w,params,stim, dtstim, lagms, wsign,rho)
            rel[j] = runOne(params, stim, correlated=True)

        # compute summary statistics and distance
        if datadim ==2:
            '''
            # old version, works if no std is included
            for j in range(0,len(gData)):
                simSS = makeSS(rel,g,gData[j,:]) # rel can be multidimensional! gData[...] is one dimensional
                #  compute dist and take mean
                dSamps[i] = dSamps[i] + (scipy.spatial.distance.euclidean(simSS,dataSS[j,:]))/len(gData)
            '''
            simSS = makeSS(rel, g, gData,w)  # both multidimensional. simSS mean in all dims
            #for j in range(0, len(gData)):
            dSamps[i] = scipy.spatial.distance.euclidean(simSS,dataSS)

        else:
            simSS = makeSS(rel,g,gData,w) # rel and gData can be multidimensional!
            dSamps[i] = scipy.spatial.distance.euclidean(simSS,dataSS[j,:])

    # stack together the samples and the distances
    outSamps = np.vstack((sampsParams,dSamps))
    outSamps = outSamps[:,outSamps[-1,:].argsort()]

    return outSamps

# has to be in the jupyter notebook, otherwise some parameters are unknwon...
'''
def run_parallel_w(*parameters):
    samps = np.array(parameters)
    #print(params)
    #print(np.shape(samps))
    outUn =  runManyNonNewW(stochStim,samps,g,trueG,dataSS,stimT,c,a,phi,params)    
    return outUn


def run_parallel_ribbon(*parameters):
    sampsParams = np.array(parameters)
    #print(np.shape(samps))
    outUn =  runManyNonNewR(stochStim,w,g,trueG,dataSS,stimT,c,a,phi,sampsParams) 
    
    return outUn
'''




##########################################################################################################


def pairDist(in1,in2,g1,g2):
    rD = scipy.spatial.distance.euclidean(g1,g2)
    sumD = sqrt((np.sum(in1)-np.sum(in2))**2)
    ssVec = [rD,sumD]
    return ssVec


def makeSS(traces, g, gData, w):
    """
    computes summary statistics of traces and returns the mean SS
    only this should be changed if new SS are involved
    ! for rDistance the SS of data is 0
    # include possible scaling for influence of single SS here!
    ---
    :param traces: simulated traces [ntraces, tpts]
    :param g: (gaussian) kernel
    :param gData: convolved data data*g [ntraces, tpts]
    :param w weights for the summary stats
    ---
    :return: summary stats of traces (mean, one-dim array with len=dimOfSS)
    """
    # specify dimension of SS
    dimOfSS = len(w)

    # specify weight vector for summary stats
    #w = np.ones(dimOfSS)
    #w[0] = (1/48)* 1 # 10 #1
    #w[1] = (1/718.25) * 10
    #w[2] = (1/124) *20 # 10
    #w[3] = (1/50.5) * 10 # 10
    #w[4] = (1/26.75) *5 # 10
    #w[5] = (1/29) *5
    #w[6] = (1/24.25) *5
    #w[7] = (1/168) *0 #10

    # deal with one and two dimensional
    ntraces = 1
    datadim = len(np.shape(traces))

    # for mutliple traces
    if datadim == 2:
        ntraces = len(traces)
        ssVecAll = np.zeros((ntraces, dimOfSS))
        for i in range(ntraces):
            ssVecAll[i] = [rDistance_mult(traces[i], g, gData),
                           sumRel(traces[i]),
                           countEvents(traces[i], 1),
                           countEvents(traces[i], 2),
                           countEvents(traces[i], 3),
                           countEvents(traces[i], 4),
                           countEvents(traces[i], 5),
                           countEvents(traces[i], 6),
                           0] # dummy for std
        ssVec = np.mean(ssVecAll, axis=0)
        ssVec[-1] = np.sum(np.std(ssVecAll, axis=0))

    # for one trace:
    else:
        ssVec = [rDistance_mult(traces, g, gData),
                 sumRel(traces),
                 countEvents(traces, 1),
                 countEvents(traces, 2),
                 countEvents(traces, 3),
                 countEvents(traces, 4),
                 countEvents(traces, 5),
                 countEvents(traces, 6),
                 0] #std: 0 if only one trace

    # possible scaling of ssVec (to change influence of single SS)
    return ssVec * w


def rDistance1(rel, g, gData):
    """
    computes the distance of conv(rel, g) and gData
    for tow 1d arrays
    ---
    :param rel: release trace of ribbon
    :param g: (gaussian) kernel
    :param gData: convolved traces of data,g
    ---
    :return: euclidean distance(gData, g*rel)
    """

    rel = np.ravel(rel)
    gRel = scipy.signal.convolve(rel, g)
    gRel = gRel[:len(gData)]
    return scipy.spatial.distance.euclidean(gData, gRel)


def rDistance_mult(trace, g, gData):
    """
    computes the mean distance of conv(trace, g) and gData
    for 1d trace, and 1 or multiple gData traces
    """
    datadim = len(np.shape(gData))

    if len(np.shape(trace)) == 2:
        warnings.warn('Problem in rDistance_mult. not all traces were used. make sure that dim(trace)==1')

    if datadim == 2:
        x = 0
        for i in range(np.shape(gData)[0]):
            x += rDistance1(trace, g, gData[i])

        x = x / (np.shape(gData)[0])

    else:
        x = rDistance1(trace, g, gData)

    return x


def countEvents(trace, c):
    """
    counts the c-fold events of trace
    """
    return np.sum(trace == c)


def sumRel(trace):
    """
    returns the total relesed vesicles of trace
    """
    return np.sum(trace)


    

########################################################################
"""
Priors 
initial and Updating
"""

# hyperparameter for the distribution of ribbon params


# initial priors
#dockMax = 8
#ribbonMax = 50


def makeHypersPrior():
    """
    :return: hyper parameters for the ribbon prior distributions
    """
    k_x0Hypers = [np.array([10, 0.5]), np.eye(2) * np.array([400, 0.1]), 4, 4]  # [mu0s,Lambda0, kappa0, nu0]
    dPHypers = [0.3, 0.05, 3, 3]  # [mu0, sigma0sq, kappa0, nu0]
    rlHypers = [2, 0.25]  # [k, scale(theta)] # mean=k*theta
    rhoHypers = [0.2, 0.05, 3, 3]  # [mu0, sigma0sq, kappa0, nu0]

    kernelscaleHypers = [1, 0.1, 3, 3]  # [mu0, sigma0sq, kappa0, nu0]

    hypers = [k_x0Hypers, dPHypers, rlHypers, rhoHypers, kernelscaleHypers]
    return hypers


def makeHypers(hypersOld, outR, nupdate_raw, importance_factor=1):
    """
    updates the prior distributions of the ribbon
    ---
    :param hypersOld: old hyperparameters of the ribbon distributions
    :param outR: sampled values for the ribbon parameters
    :param nupdate_raw: taking into account the n best samples (for updating)
    :param importance_factor: scaling nupdate_raw by this factor for updating rules (but not update sample size)
    ---
    :return: updated priors
    """

    nupdate = nupdate_raw * importance_factor

    # NL parameter k and x0
    ## multivariate normal distribution, 2d
    mu0s, Lambda0, kappa0, nu0 = hypersOld[0]
    mean_samples = np.mean(outR[:2, :nupdate_raw], axis=1)
    muns = 1 / (kappa0 + nupdate) * (kappa0 * mu0s + nupdate * mean_samples)
    kappan = kappa0 + nupdate
    nun = nu0 + nupdate

    # compute matrix with y_i - mean(y)
    yi_y = outR[:2, :nupdate_raw] - np.repeat(mean_samples, nupdate_raw).reshape(2, nupdate_raw)
    # compute s: sum (y - mean(y)) * (y - mean(y)).T (2x2 corr matrix * n)
    s = np.sum([yi_y[:, i].reshape(2, 1) * yi_y[:, i].reshape(1, 2) for i in range(np.shape(yi_y)[1])], axis=0)

    Lambdan = (Lambda0
               + s
               + (kappa0 * nupdate) / (kappa0 + nupdate) * np.dot((mean_samples - np.array(mu0s)).reshape(2, 1),
                                                                  (mean_samples - np.array(mu0s)).reshape(1, 2)))
    k_x0Hypers = [muns, Lambdan, kappan, nun]

    # dock P parameter
    # for normal distribution
    # [mu0, sigma0sq, kappa0, nu0]
    mu0, sigma0sq, kappa0, nu0 = hypersOld[1]
    mean_samples = np.mean(outR[2, :nupdate_raw])
    mun = 1 / (kappa0 + nupdate) * (kappa0 * mu0 + nupdate * mean_samples)
    kappan = kappa0 + nupdate
    nun = nu0 + nupdate

    s_squared = np.var(outR[2, :nupdate_raw], ddof=1)
    sigman_squared = 1 / nun * (nu0 * sigma0sq + (nupdate - 1) * s_squared + (kappa0 * nupdate) / (kappa0 + nupdate) * (
                mean_samples - mu0) ** 2)

    dPHypers = [mun, sigman_squared, kappan, nun]  # [mun, sigman^2, kappan, nun]

    # ribbon Lambda parameter
    # for Gamma distribution
    k = hypersOld[2][0] + np.sum(outR[3, :int(np.round(nupdate))])
    theta = hypersOld[2][1] / (1 + int(np.round(nupdate)) * hypersOld[2][1])
    rlHypers = [k, theta]  # [k, scale(theta)] # mean=k*theta

    # rho --- Correlation parameter
    # for normal distribution
    # [mu0, sigma0sq, kappa0, nu0]
    mu0, sigma0sq, kappa0, nu0 = hypersOld[3]
    mean_samples = np.mean(outR[6, :nupdate_raw])
    mun = 1 / (kappa0 + nupdate) * (kappa0 * mu0 + nupdate * mean_samples)
    kappan = kappa0 + nupdate
    nun = nu0 + nupdate

    s_squared = np.var(outR[6, :nupdate_raw], ddof=1)
    sigman_squared = 1 / nun * (nu0 * sigma0sq + (nupdate - 1) * s_squared + (kappa0 * nupdate) / (kappa0 + nupdate) * (
                mean_samples - mu0) ** 2)

    rhoHypers = [mun, sigman_squared, kappan, nun]  # [mun, sigman^2, kappan, nun]

    # kernelscale
    # for normal distribution
    # [mu0, sigma0sq, kappa0, nu0]
    mu0, sigma0sq, kappa0, nu0 = hypersOld[4]
    mean_samples = np.mean(outR[7, :nupdate_raw])
    mun = 1 / (kappa0 + nupdate) * (kappa0 * mu0 + nupdate * mean_samples)
    kappan = kappa0 + nupdate
    nun = nu0 + nupdate

    s_squared = np.var(outR[7, :nupdate_raw], ddof=1)
    sigman_squared = 1 / nun * (nu0 * sigma0sq + (nupdate - 1) * s_squared + (kappa0 * nupdate) / (kappa0 + nupdate) * (
            mean_samples - mu0) ** 2)

    kernelscaleHypers = [mun, sigman_squared, kappan, nun]  # [mun, sigman^2, kappan, nun]

    # stack all hypers together
    hypers = [k_x0Hypers, dPHypers, rlHypers, rhoHypers, kernelscaleHypers]

    return hypers


def drawSamps(constants, hyperMat, nSamps):
    """
    draw samples from priors  using updated hyper parameters
    including drawing of Sigma for normal distribution
    ---
    :param constants: which params to draw
    :param hyperMat: generated list of hyper parameters from makeHypers
    :param nSamps: how many samples to draw
    ---
    :return:
    """
    # Draw from (proposal) priors, if not constant. if constant, takes value from second row of constants

    # draw jointly NL params: k and x0
    # in two steps
    if constants[0, 0] == 0 and constants[0, 1] == 0:  # k not constant
        # define hard cut offs
        cuts_k = [0, 50]
        cuts_x0 = [-2, 3]  # scale with the Ca range

        muns, Lambdan, kappan, nun = hyperMat[0]

        sigmas = invwishart(df=nun, scale=Lambdan).rvs(
            size=nSamps)  # take care scale is here the "precision" E[~] = ...*Lambdan

        draws = np.array([scp.stats.multivariate_normal(mean=muns, cov=sigmas[i]).rvs() for i in range(nSamps)])
        kSamps = draws[:, 0]
        x0Samps = draws[:, 1]

        while (sum(kSamps < cuts_k[0]) + sum(kSamps > cuts_k[1]) + sum(x0Samps < cuts_x0[0]) + sum(
                x0Samps > cuts_x0[1])) != 0:
            for i in range(0, int(nSamps)):
                if kSamps[i] < cuts_k[0] or kSamps[i] > cuts_k[1] or x0Samps[i] < cuts_x0[0] or x0Samps[i] > cuts_x0[1]:
                    sigma1 = invwishart(df=nun,
                                        scale=Lambdan).rvs()  # take care scale is here the "precision" E[~] = ...*Lambdan
                    draw1 = np.array(scp.stats.multivariate_normal(mean=muns, cov=sigma1).rvs())
                    kSamps[i] = draw1[0]
                    x0Samps[i] = draw1[1]

    else:
        x0Samps = np.ones(int(nSamps)) * constants[1, 1]
        kSamps = np.ones(int(nSamps)) * constants[1, 0]

    if constants[0, 2] == 0:  # dockP not constant
        # define hard cutoff
        cut_dP = [0, 1]  # 0,1

        # get the updated hyper parameters
        mun, sigmansq, kappan, nun = hyperMat[1]

        # sample in 2 steps:
        # 1. sample sigma (for this sample from chi2 and invert and scale)
        # 2. sample from normal with this sigma
        normChi = chi2.rvs(nun, size=nSamps)
        invScaledChi = nun * sigmansq / normChi  # E = nu/(nu-2) *sQ !!! for Var not STD!!!
        stdNew = np.sqrt(invScaledChi)

        dockPSamps = norm.rvs(loc=mun, scale=stdNew, size=int(nSamps))

        while (sum(dockPSamps < cut_dP[0]) + sum(dockPSamps > cut_dP[1])) != 0:
            for i in range(0, int(nSamps)):
                if dockPSamps[i] < cut_dP[0] or dockPSamps[i] > cut_dP[1]:
                    normChi = chi2.rvs(nun)
                    invScaledChi = nun * sigmansq / normChi  # E = nu/(nu-2) *sQ !!! for Var not STD!!!
                    stdNew = np.sqrt(invScaledChi)
                    dockPSamps[i] = norm.rvs(loc=mun, scale=stdNew)
    else:
        dockPSamps = np.ones(int(nSamps)) * constants[1, 2]


    if constants[0, 3] == 0:  # ribbonLambda not constant
        # hard cut off
        cut_lambda = [0, 1]  # 0,1

        kLambda = hyperMat[2][0]
        thetaLambda = hyperMat[2][1]

        # ribbonLambdaSamps = beta.rvs(hyperMat[3,0],hyperMat[3,1],size = int(nSamps))
        # ribbonLambdaSamps= uniform.rvs(0,.5,size=int(nSamps))
        ribbonLambdaSamps = gamma.rvs(kLambda, scale=thetaLambda, size=int(nSamps))
        while (sum(ribbonLambdaSamps < cut_lambda[0]) + sum(ribbonLambdaSamps > cut_lambda[1])) != 0:
            for i in range(0, len(ribbonLambdaSamps)):
                if ribbonLambdaSamps[i] < cut_lambda[0] or ribbonLambdaSamps[i] > cut_lambda[1]:
                    ribbonLambdaSamps[i] = gamma.rvs(kLambda, scale=thetaLambda)
    else:
        ribbonLambdaSamps = np.ones(int(nSamps)) * constants[1, 3]

    dockMax = np.ones(nSamps) * constants[1, 4]
    ribbonMax = np.ones(nSamps) * constants[1, 5]

    if constants[0, 6] == 0:  # rho not constant
        # define hard cutoff
        cut_rho = [0, 1]  # 0,1

        # get the updated hyper parameters
        mun, sigmansq, kappan, nun = hyperMat[3]

        # sample in 2 steps:
        # 1. sample sigma (for this sample from chi2 and invert and scale)
        # 2. sample from normal with this sigma
        normChi = chi2.rvs(nun, size=nSamps)
        invScaledChi = nun * sigmansq / normChi  # E = nu/(nu-2) *sQ !!! for Var not STD!!!
        stdNew = np.sqrt(invScaledChi)

        rhoSamps = norm.rvs(loc=mun, scale=stdNew, size=int(nSamps))

        while (sum(rhoSamps < cut_rho[0]) + sum(rhoSamps > cut_rho[1])) != 0:
            for i in range(0, int(nSamps)):
                if rhoSamps[i] < cut_rho[0] or rhoSamps[i] > cut_rho[1]:
                    normChi = chi2.rvs(nun)
                    invScaledChi = nun * sigmansq / normChi  # E = nu/(nu-2) *sQ !!! for Var not STD!!!
                    stdNew = np.sqrt(invScaledChi)
                    rhoSamps[i] = norm.rvs(loc=mun, scale=stdNew)
    else:
        rhoSamps = np.ones(int(nSamps)) * constants[1, 6]


    if constants[0, 7] == 0:  # kernelscale not constant
        # define hard cutoff
        cut_kernelscale = [0.05, 2]  # 0,1

        # get the updated hyper parameters
        mun, sigmansq, kappan, nun = hyperMat[4]

        # sample in 2 steps:
        # 1. sample sigma (for this sample from chi2 and invert and scale)
        # 2. sample from normal with this sigma
        normChi = chi2.rvs(nun, size=nSamps)
        invScaledChi = nun * sigmansq / normChi  # E = nu/(nu-2) *sQ !!! for Var not STD!!!
        stdNew = np.sqrt(invScaledChi)

        kernelscaleSamps = norm.rvs(loc=mun, scale=stdNew, size=int(nSamps))

        while (sum(kernelscaleSamps < cut_kernelscale[0]) + sum(kernelscaleSamps > cut_kernelscale[1])) != 0:
            for i in range(0, int(nSamps)):
                if kernelscaleSamps[i] < cut_kernelscale[0] or kernelscaleSamps[i] > cut_kernelscale[1]:
                    normChi = chi2.rvs(nun)
                    invScaledChi = nun * sigmansq / normChi  # E = nu/(nu-2) *sQ !!! for Var not STD!!!
                    stdNew = np.sqrt(invScaledChi)
                    kernelscaleSamps[i] = norm.rvs(loc=mun, scale=stdNew)
    else:
        kernelscaleSamps = np.ones(int(nSamps)) * constants[1, 7]

    samps = np.stack((kSamps, x0Samps, dockPSamps, ribbonLambdaSamps, dockMax, ribbonMax, rhoSamps, kernelscaleSamps))
    return samps



############################################################################


def Calc_VarSSData(traces, gData, g):
    """
    calculates the mean and var of the summary stats of the data
    ---
    pairwise comparison
    """
    onWhich = 0
    nTrue = len(traces)
    nPairs = int((nTrue * (nTrue - 1)) / 2)
    dataDists = np.zeros(nPairs)
    for i in range(0, nTrue):
        for j in range(0, nTrue):
            if i < j:
                dataDists[onWhich] = pairwise_dist(traces[i], traces[j], gData[i], gData[j], g)
                onWhich = onWhich + 1

    trueMean = np.mean(dataDists)
    trueVar = np.var(dataDists)
    return trueMean, trueVar, dataDists


def pairwise_dist(trace1, trace2, g1, g2, g):
    SS1 = makeSS(trace1, g, g2)
    SS2 = makeSS(trace2, g, g1)
    return scipy.spatial.distance.euclidean(SS1, SS2)


def Calc_VarSS_fit(traces, dataSS, trueG,g, w, mode='fitted'):
    """
    calculates the mean and var of the summary stats of traces to the meanSS
    ---
    :param w weights for sumstats
    """
    meandataSS = np.mean(dataSS, axis=0)

    fitSS = np.zeros((len(traces), len(trueG), 8))

    for i in range(len(traces)):
        for j in range(len(trueG)):
            fitSS[i, j] = makeSS(traces[i], g, trueG[j],w)

    fitSS1 = np.mean(fitSS, axis=1)

    # print(fitSS1)
    dists = np.zeros(len(fitSS1))
    for i in range(len(fitSS1)):
        dists[i] = scipy.spatial.distance.euclidean(fitSS1[i], meandataSS)

    meanDist = np.mean(dists)
    varDist = np.var(dists)

    return meanDist, varDist, dists
