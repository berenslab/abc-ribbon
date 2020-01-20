import numpy as np
import scipy as scp
import scipy.stats



# updating prior distribution
def makeHypers(hypersOld, params, nupdate):
    """
    updates the prior distributions of the ribbon
    ---
    :param hypersOld: old hyperparameters of the ribbon distributions
    :param params: sampled parameters, already sorted by loss. shape: (n_samples, dim)
    :param nupdate_raw: taking into account the n best samples (for updating)
    ---
    :return: updated priors
    """

    ## multivariate normal distribution
    mu0s, Lambda0, kappa0, nu0 = hypersOld

    dim = len(mu0s) # dimension of parameter space

    mean_samples = np.mean(params[:nupdate], axis=0)
    muns = 1 / (kappa0 + nupdate) * (kappa0 * mu0s + nupdate * mean_samples)
    kappan = kappa0 + nupdate
    nun = nu0 + nupdate

    # compute matrix with y_i - mean(y)
    yi_y = params[:nupdate] - np.tile(mean_samples, nupdate).reshape(nupdate,dim)
    # compute s: sum (y - mean(y)) * (y - mean(y)).T (dim x dim corr matrix * n)
    s = np.sum([yi_y[0].reshape(dim, 1) *yi_y[0].reshape(1, dim) for i in range(len(yi_y))], axis=0)

    Lambdan = (Lambda0
               + s
               + (kappa0 * nupdate) / (kappa0 + nupdate) * np.dot((mean_samples - np.array(mu0s)).reshape(dim, 1),
                                                                  (mean_samples - np.array(mu0s)).reshape(1, dim)))
    hypers = [muns, Lambdan, kappan, nun]

    return hypers


# draw samples
def drawSamps(hyperMat, nSamps):
    """
    draw samples from priors  using updated hyper parameters
    including drawing of Sigma for normal distribution
    ---
    :param hyperMat: parameters from makeHypers
    :param nSamps: how many samples to draw
    ---
    :return:
    """

    # in two steps

    # define hard cut offs
    # not yet implemented
    #cuts_0 = [0, 50]
    #cuts_1 = [-2, 3]  

    muns, Lambdan, kappan, nun = hyperMat

    sigmas = scipy.stats.invwishart(df=nun, scale=Lambdan).rvs(
        size=nSamps)  # take care scale is here the "precision" E[~] = ...*Lambdan

    draws = np.array([scp.stats.multivariate_normal(mean=muns, cov=sigmas[i]).rvs() for i in range(nSamps)])
    
    '''
    # code snippet to truncate the (draws from the) distribution
    # param_labels = ['k','x0', 'refill RRP', 'refill IP', 'RRP', 'IP', 'rho', 'kernelscale']
    boundaries = np.zeros((8,2))
    boundaries[0] = [1,np.inf]
    boundaries[1] = [-5,5]
    boundaries[2] = [0,np.inf]
    boundaries[3] = [0,np.inf]
    boundaries[4] = [0,np.inf]
    boundaries[5] = [0,np.inf]
    boundaries[6] = [0.01,0.99]
    boundaries[7] = [0.01,np.inf]

    while not( (draws > boundaries[:,0]).all()  and (draws < boundaries[:,1]).all()):
        for i in range(0, int(nSamps)):
            if not( (draws[i] > boundaries[:,0]).all() and (draws[i] < boundaries[:,1]).all()):
                sigma1= scipy.stats.invwishart(df=nun, scale=Lambdan).rvs(
                            size=1)  # take care scale is here the "precision" E[~] = ...*Lambdan

                draw1 = scp.stats.multivariate_normal(mean=muns, cov=sigma1).rvs()
                draws[i] = draw1
    '''

    return draws
