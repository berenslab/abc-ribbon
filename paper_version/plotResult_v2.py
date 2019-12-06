import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
import  scipy as scp
import scipy
from matplotlib import gridspec
#from rejNew import *
#from ribbon import *

"""
raster plot to show raw traces
"""


def plot_raster1(trace, dt=0.01, color='b', offset=0, figsize=(12, 8)):
    """
    plots a raster of one trace, centered at offset
    ---
    trace: 1d trace containing the quantal events
    dt: time resolution of trace in sec
    """
    time = np.arange(0, len(trace)) * dt
    timepts = time[trace != 0]
    amps = trace[trace != 0]
    plt.figure(1, figsize=figsize)
    plt.vlines(timepts, -amps / 2 + offset, amps / 2 + offset, color=color)
    plt.xlabel('sec')
    plt.yticks([])


def plot_raster1_new(ax, trace, dt=0.01, color='b', offset=0, label=None, scalebar=False, scalebarcolor='b', 
                     normalize=True, upper=7):
    """
    plots a raster of one trace, centered at offset
    ---
    trace: 1d trace containing the quantal events
    dt: time resolution of trace in sec
    normalize: all data is alway plotted to y-axis [0,upper]
    """
    time = np.arange(0, len(trace)) * dt
    timepts = time[trace != 0]
    amps = trace[trace != 0]
    # plt.figure(1, figsize=figsize)
    # ax=plt.subplot()

    # centered version
    # ax.vlines(timepts, -amps / 2 + offset, amps / 2 + offset, color=color, label=label)

    # version at baseline
    ax.vlines(timepts, 0, amps + offset, color=color, label=label)

    # plot scalebar
    if scalebar:
        ax.vlines(time[-3], 0, 5, color=scalebarcolor, linewidths=2)
        plt.text(time[-3] + 0.3, 2, '5 vesicles')

    plt.xlim(time[0], time[-1])
    # plt.xlabel('sec')
    plt.yticks([])
    # plt.legend()
    plt.xticks([])
    # sns.despine()
    
    if normalize:
        plt.ylim(0,upper)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

def plot_rastern(traces, dt=0.01, color='b', figsize=(12, 8), offsetpertrace=10, totaloffset=0):
    """
    plots multiple raster1 plots
    ---
    traces: 2d array containing the quantal events [n, tpts]
    offsetpertrace: offset between traces, should be something like max. amplitude+1
    totaloffset: if two plots are generated this could be used to seperate the two groups of rasterplots
    """
    for i in range(len(traces)):
        plot_raster1(traces[i], dt=dt, color=color, offset=i * offsetpertrace + totaloffset, figsize=figsize)



""""
functions from model
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
    y = 1/(1+np.exp(-k*(x-x0)))
    return y


"""
Plot fcts for the density/histogram plots for ribbon params and weights
"""


# defining an own kde fct to add points for true values
def kde_self(data, data2=None, shade=False, vertical=False,
             kernel='gau', bw='scott', gridsize=100, cut=3, clip=None, legend=True, cumulative=False, shade_lowest=True,
             cbar=False, cbar_ax=None, cbar_kws=None, ax=None, **kwargs):
    sns.kdeplot(data[:-1], data2[:-1], cmap="Blues_d")
    if len(data.values) == 1:
        plt.plot(data.values[-1], data2.values[-1], 'o', color='r')


def hist_self(data, bins=None, range=None, density=None, weights=None, cumulative=False, bottom=None,
              histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False,
              color=None, label=None, stacked=False, normed=None, hold=None, **kwargs):
    if len(data) != 1:
        sns.distplot(data, kde=False)
    if len(data) == 1:
        plt.axvline(data[-1], color='r')
    # plt.axvline(dFrame.get_value(lastindex, x.name),**kwargs)


def plot_kde_samples(kSamps, x0Samps, dockPSamps, ribbonLambdaSamps, rhoSamps, kernelscaleSamps,
                     TrueK=None, Truex0=None, TrueribbonLambda=None, TruedockP=None, TrueRho=None, TrueKernelscale=None,
                     plot_true=False, markersize=30):
    if plot_true == True:
        # append true value in each vector
        kSamps = np.hstack((kSamps, TrueK))
        x0Samps = np.hstack((x0Samps, Truex0))
        dockPSamps = np.hstack((dockPSamps, TruedockP))
        ribbonLambdaSamps = np.hstack((ribbonLambdaSamps, TrueribbonLambda))
        rhoSamps = np.hstack((rhoSamps, TrueRho))
        kernelscaleSamps = np.hstack((kernelscaleSamps, TrueKernelscale))


        # define markers: 0 for samples, 1 for truevalues
        marker = np.zeros(len(kSamps))
        marker[-1] = 1
        lastindex = len(kSamps) - 1


    else:
        # define markers
        marker = np.zeros(len(kSamps))
        lastindex = len(kSamps) - 1

    dSetFrame = {"k": kSamps, "x0": x0Samps, "dockP": dockPSamps,
                 "ribbonLambda": ribbonLambdaSamps, "rho": rhoSamps,'kernelscale':kernelscaleSamps, "marker": marker}
    dFrame = pd.DataFrame(dSetFrame)
    plt.figure(1)
    g = sns.PairGrid(dFrame, hue='marker', vars=["k", "x0", "dockP", "ribbonLambda", "rho", "kernelscale"],
                     palette={0: sns.xkcd_rgb["windows blue"], 1: sns.xkcd_rgb["pale red"]})
    g = g.map_upper(plt.scatter, s=markersize)
    # g = g.map_lower(plt.scatter, s=markersize)
    # g = g.map_lower(sns.kdeplot, cmap="Blues_d")
    g = g.map_lower(kde_self, cmap="Blues_d")
    # g = g.map_diag(plt.hist)#, histtype='step', linewidth=3)
    # g = g.map_diag(lambda x, **kwargs: plt.axvline(dFrame.get_value(lastindex, x.name),**kwargs))
    g = g.map_diag(hist_self)

    plt.show()
###########################################################################
# plotting time course
def get_expected_values(hypers):
    """
    returns expected values for one set of hyper params
    """
    expected_values = np.zeros(6)

    expected_values[0] = hypers[0] [0][0]
    expected_values[1] = hypers[0] [0][1]
    expected_values[2] = hypers[1] [0]
    expected_values[3] = hypers[2] [0] * hypers[2][1]
    expected_values[4] = hypers[3] [0]
    expected_values[5] = hypers[4] [0]

    return expected_values

def get_expected_values_all(allhypers, hypers_prior):
    """
    returns expected values for a list of sets of hyper params
    """
    expected_values = np.zeros((len(allhypers) + 1, 6))
    expected_values[0] = get_expected_values(hypers_prior)
    for i in range(1, len(expected_values)):
        expected_values[i] = get_expected_values(allhypers[i - 1])

    return expected_values




def get_stds(hypers):
    """
    returns expected values for one set of hyper params
    !!! gamma distriburions is not symmetric
    """
    stds = np.zeros(6)

    stds[0] = (1/(hypers[0][3] -2) * hypers[0][1][0,0] )**0.5 #  [muns,Lambdan, kappan, nun]
    stds[1] = (1/(hypers[0][3] -2) * hypers[0][1][1,1])**0.5 #  [muns,Lambdan, kappan, nun]
    stds[2] = (hypers[1][3]/(hypers[1][3]-2)* hypers[1][1] )**0.5  # [mun, sigman^2, kappan, nun]
    stds[3] = (hypers[2][0] * hypers[2][1]**2)**0.5
    stds[4] = (hypers[3][3]/(hypers[3][3]-2)* hypers[3][1] )**0.5  # [mun, sigman^2, kappan, nun]
    stds[5] = (hypers[4][3]/(hypers[4][3]-2)* hypers[4][1] )**0.5  # [mun, sigman^2, kappan, nun]


    return stds

def get_stds_all(allhypers, hypers_prior):
    """
    returns expected values for a list of sets of hyper params
    """
    stds = np.zeros((len(allhypers) + 1, 6))
    stds[0] = get_stds(hypers_prior)
    for i in range(1, len(stds)):
        stds[i] = get_stds(allhypers[i - 1])

    return stds


def plot_time_course(outRSave, exp_values, nbest=100, plotTrue=False, trueParams=None, plotStds=False, stds=None,
                     ylimits=[None]*6,
                     param_labels=['K', 'X0', 'DockP', 'RibbonLambda', 'rho', 'kernelscale']):
    runs = np.shape(outRSave)[0]

    # calculate mean of best params per run (still includes the fixed params)
    means_raw = np.zeros((runs, 9))
    for i in range(runs):
        means_raw[i] = np.mean(outRSave[i][:, :nbest], axis=1)

    means = np.zeros((runs, 6))
    means[:, :4] = means_raw[:, :4]
    means[:, 4] = means_raw[:, 6]
    means[:, 5] = means_raw[:, 7]


    plt.figure(1, figsize=(12, 8))
    plt.suptitle('Time course of the expected values of (proposal) prior', y=1.01)
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.title(param_labels[i])
        plt.plot(means[:, i], label='mean of best params')
        plt.plot(exp_values[:, i], label='expected values from (proposal) prior')
        if plotTrue:
            plt.axhline(trueParams[i], color='red', alpha=0.5, label='True value')
        if plotStds:
            plt.fill_between(np.arange(len(exp_values)),exp_values[:,i]-2*stds[:,i],
                             exp_values[:,i]+2*stds[:,i],
                             alpha=0.3,
                             color='b', label='+- 2std')
        plt.ylim(ylimits[i])

        plt.xlabel('round')
    plt.tight_layout()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#
#hypers_prior = makeHypersPrior()
#exp_values = get_expected_values_all(hypersSave, hypers_prior)
#stds = get_stds_all(hypersSave, hypers_prior)
#trueParams = [trueK, trueX0, trueDockP, trueRibbonLambda,truerho ]
#plot_time_course(outRSave, exp_values, plotTrue=True, trueParams=trueParams, plotStds=True, stds=stds,
#                ylimits=[[0,50],[0.25,1.25],[0.275,0.4],[0.1,0.6],[0.1,0.8]])
#
############################
"""
plot 1D marginals
"""

def plot_1d_marginals(hypers,
                      label=None,
                      limits=[[0, 50], [0, 2], [0, 1], [0, 1], [0, 1],[0,2]],
                      param_labels=['K', 'X0', 'DockP', 'RibbonLambda', 'rho', 'kernelscale'],
                      plot_true=False,
                      truevalues=None,
                      figsize=(12, 6)):
    """
    :param hypers: hyperparameter of one round
    :param label:
    :param limits:
    :param param_labels:
    :param plot_true:
    :param truevalues:
    :param figsize:
    :return:
    """
    mus_normal = np.zeros(6)
    mus_normal[0] = hypers[0][0][0]
    mus_normal[1] = hypers[0][0][1]
    mus_normal[2] = hypers[1][0]
    mus_normal[3] = -1  # no normal distribution
    mus_normal[4] = hypers[3][0]
    mus_normal[5] = hypers[4][0]

    stds_normal = np.zeros(6)
    stds_normal[0] = (1 / (hypers[0][3] - 3) * hypers[0][1][0, 0] ) ** 0.5  # [muns,Lambdan, kappan, nun]
    stds_normal[1] = (1 / (hypers[0][3] - 3) * hypers[0][1][1, 1] ) ** 0.5  # [muns,Lambdan, kappan, nun]
    stds_normal[2] = (hypers[1][3] / (hypers[1][3] - 2) * hypers[1][1] ) ** 0.5  # [mun, sigman^2, kappan, nun]
    stds_normal[3] = -1  # no normal distribution
    stds_normal[4] = (hypers[3][3] / (hypers[3][3] - 2) * hypers[3][1] ) ** 0.5  # [mun, sigman^2, kappan, nun]
    stds_normal[5] = (hypers[4][3] / (hypers[4][3] - 2) * hypers[4][1] ) ** 0.5  # [mun, sigman^2, kappan, nun]

    # for ribbon lambda, gamma distribution
    kLambda = hypers[2][0]
    thetaLambda = hypers[2][1]

    plt.figure(1, figsize=figsize)
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.title(param_labels[i])
        x = np.arange(limits[i][0], limits[i][1], 0.001)
        if not (i == 3):
            y = scp.stats.norm.pdf(x, mus_normal[i], stds_normal[i])
        else:
            y = scp.stats.gamma.pdf(x, kLambda, scale=thetaLambda)

        plt.plot(x, y, label=label)
        if plot_true:
            plt.axvline(truevalues[i], color='r', alpha=0.5)
    plt.tight_layout()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


##########################

"""
plot 2 D marginals
"""

def make_posteriorlist(hypers):
    """
    returns posteriors.
    1d for all,
    last element 2d posterior for first two params
    """
    posteriors = [[] for _ in range(7)]

    mus_normal = np.zeros(6)
    mus_normal[0] = hypers[0][0][0]
    mus_normal[1] = hypers[0][0][1]
    mus_normal[2] = hypers[1][0]
    mus_normal[3] = -1  # no normal distribution
    mus_normal[4] = hypers[3][0]
    mus_normal[5] = hypers[4][0]


    stds_normal = np.zeros(6)
    stds_normal[0] = (1 / (hypers[0][3] - 3) * hypers[0][1][0, 0]) ** 0.5  # [muns,Lambdan, kappan, nun]
    stds_normal[1] = (1 / (hypers[0][3] - 3) * hypers[0][1][1, 1]) ** 0.5  # [muns,Lambdan, kappan, nun]
    stds_normal[2] = (hypers[1][3] / (hypers[1][3] - 2) * hypers[1][1]) ** 0.5  # [mun, sigman^2, kappan, nun]
    stds_normal[3] = -1  # no normal distr
    stds_normal[4] = (hypers[3][3] / (hypers[3][3] - 2) * hypers[3][1]) ** 0.5  # [mun, sigman^2, kappan, nun]
    stds_normal[5] = (hypers[4][3] / (hypers[4][3] - 2) * hypers[4][1]) ** 0.5  # [mun, sigman^2, kappan, nun]

    cov01 = 1 / (hypers[0][3] - 3) * hypers[0][1]  # [muns,Lambdan, kappan, nun]

    # for ribbon lambda, gamma distribution
    kLambda = hypers[2][0]
    thetaLambda = hypers[2][1]

    posteriors[0] = lambda x: scp.stats.norm.pdf(x, mus_normal[0], stds_normal[0])
    posteriors[1] = lambda x: scp.stats.norm.pdf(x, mus_normal[1], stds_normal[1])
    posteriors[2] = lambda x: scp.stats.norm.pdf(x, mus_normal[2], stds_normal[2])
    posteriors[3] = lambda x: scp.stats.gamma.pdf(x, kLambda, scale=thetaLambda)
    posteriors[4] = lambda x: scp.stats.norm.pdf(x, mus_normal[4], stds_normal[4])
    posteriors[5] = lambda x: scp.stats.norm.pdf(x, mus_normal[5], stds_normal[5])

    posteriors[-1] = lambda x1, x2: scp.stats.multivariate_normal.pdf([x1, x2], mean=[mus_normal[0], mus_normal[1]],
                                                                     cov=cov01)
    return posteriors


def eval_grid_posterior(posterior, bounds1, bounds2, tpts):
    """
    evaluate 2 dim pdf on a grid [bounds]x[bounds2]=[lower_i,upper_i]^2 with tpts
    posterior: 2dim pdf
    bounds1 lower and upper bound for first dim
    bounds2: lower and upper bound for first dim
    tpts: number of points in each direction
    ----
    returns:
    pij: evaluated pdf
    t1,t2 : axis in both direction
    """

    # generating the grid
    t1 = np.linspace(bounds1[0], bounds1[1], tpts)
    t2 = np.linspace(bounds2[0], bounds2[1], tpts)

    # evaluating pdf on grid
    pij = np.zeros((tpts, tpts))

    for i in range(tpts):
        for j in range(tpts):
            pij[i, j] = posterior(t1[i], t2[j])

    return pij, t1, t2

def evaluate_2D_marginals(posteriors, bounds, tpts):
    """
    create df with the 2D marginals and 1D marginals on diagonal
    :param
        posteriors: 2d posteriors/list of 1d
        bounds: list of 6,2, with bounds lower and upper for each dimension
                exp: [[10,30],[0,1],[0,1],[0,1],[0,1],[0,2]]
        tpts: int, number of tpts to evaluate, evenly spaced between upper and lower
    ---
    :return
        marginal2D_df: panda df
        ts: list of time axis
    """
    marginal2D_df = [[[] for i in range(6)] for _ in range(6)]

    for i in range(6):
        for j in range(6):
            if i < j:
                if i == 0 and j == 1:
                    posterior = posteriors[-1]
                else:
                    posterior = lambda x1, x2: posteriors[i](x1) * posteriors[j](x2)
                pij, t1, t2 = eval_grid_posterior(posterior, bounds[i], bounds[j], tpts)

                # create dataframe
                df = pd.DataFrame(pij[::], columns=t2)
                df.insert(loc=0, column='t1', value=t1, allow_duplicates=False)
                df.set_index('t1')
                df = df.set_index('t1')
                marginal2D_df[i][j] = df

    # 1D margins on diagonal
    ts = [[] for _ in range(6)]
    for i in range(6):
        ts[i] = np.linspace(bounds[i][0], bounds[i][1], tpts)
        posterior = posteriors[i]
        marginal2D_df[i][i] = posterior(ts[i])

    return marginal2D_df, ts


# the plotting fcts

# the plotting fcts
def plot_2d_marginals(marginal2D_df, ts,
                      label1d='posterior',
                      plot_points=False, points=None,
                      plot_prior=False, marginal2D_df_prior=None,
                      cmap='magma', color='blue',
                      plotcontours=False, contours=None, contourcolors=['yellow', 'orange'],
                      # contours not yet implemented
                      plot2dprior=False, priorcontours=None, priorcontouralpha=0.2,
                      plot_true=False, truepoint=None,
                      plot_hist=False):
    """
    plots all 2d marginals
    ---
    :param marginal2D_df: pandas df, output[0] of evaluate_2D_marginals
    :param t_marginal: output[1] of evaluate_2D_marginals
    :param boundsnorm: normalization bounds to scale 1d marginals
    :param plotpoints: boolean, if True: plot points into 2D marginals
    :param points: list, list of points to plot
    :return:
    """
    # set seaborn
    sns.set_context("notebook")
    sns.set_style("white")

    labels = param_labels=['$k$', '$h$', '$p_r$', '$\\lambda_c$', '$\\rho$', '$\\gamma$']
    #['K', 'X0', 'DockP', 'RibbonLambda', 'rho', 'kernelscale']

    # plotting
    fig = plt.figure(1, figsize=(20, 20))
    count = 0

    for i in range(6):
        for j in range(6):
            count += 1

            if i < j:
                plt.subplot(6, 6, count)

                ax = sns.heatmap(marginal2D_df[i][j][::-1],
                                 xticklabels=[],
                                 yticklabels=[],
                                 square=True, cbar=False,
                                 linewidths=0, linecolor='black',
                                 rasterized=True,
                                 cmap=cmap,
                                 )
                plt.ylabel('')
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)

                if plot_points:
                    for nrpoint in range(len(points)):
                        # plotting single points
                        singlepoint = points[nrpoint]
                        # print(singlepoint[i],singlepoint[j])

                        # normalize to the size of datafram (st all values are between [0,1] * size(df))
                        xi_scaled = (singlepoint[i] - ts[i][0]) / (ts[i][-1] - ts[i][0]) * len(ts[i])

                        xj_scaled = (singlepoint[j] - ts[j][0]) / (ts[j][-1] - ts[j][0]) * len(ts[j])
                        plt.plot(xj_scaled, len(ts[i]) - xi_scaled, 'o', color=color, alpha=0.5)

                        # plt.axvline(xj_scaled)
                        # plt.axhline(xi_scaled)
                if plot_true:
                    # plotting single points
                    singlepoint = truepoint

                    # normalize to the size of datafram (st all values are between [0,1] * size(df))
                    xi_scaled = (singlepoint[i] - ts[i][0]) / (ts[i][-1] - ts[i][0]) * len(ts[i])

                    xj_scaled = (singlepoint[j] - ts[j][0]) / (ts[j][-1] - ts[j][0]) * len(ts[j])
                    plt.plot(xj_scaled, len(ts[i]) - xi_scaled, 'o', color='r', alpha=1)

                    # plt.axvline(xj_scaled)
                    # plt.axhline(xi_scaled)
                if plotcontours:
                    plt.contour(contours[i, j], colors=contourcolors, alpha=0.9)

                if plot2dprior:
                    plt.contour(priorcontours, colors=['orange', 'orange'], alpha=priorcontouralpha)


            elif i == j:
                ax = plt.subplot(6, 6, count)
                plt.title(labels[i])

                # plot 1d marginals
                current_color = 'b'#sns.color_palette()[1]
                plt.plot(ts[i], marginal2D_df[i][i],
                         color=current_color,
                         label=label1d)

                if plot_prior:
                    # plot one 1D prior
                    plt.plot(ts[i], marginal2D_df_prior[i][i], label='prior', color='darkorange')

                if plot_true:
                    plt.axvline(truepoint[i], color='r', label='true value')

                if plot_hist:
                    sns.distplot(points[:, i], kde=True)

                plt.xlim(ts[i][0], ts[i][-1])
                ax.spines['left'].set_visible(True)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(True)  # plot legend in last case
                if i == 5:
                    plt.legend(loc='center left', bbox_to_anchor=(-1, 0.5))

                # remove y ticks
                plt.yticks([])

    # plt.tight_layout()

    fig.subplots_adjust(top=0.95)
    # fig.suptitle('This is an amazing title')



def extract_points(outR, npoints):
    points = np.zeros((npoints, 6))
    points[:, :4] = outR[:4, :npoints].T
    points[:, 4] = outR[6, :npoints]
    points[:, 5] = outR[7, :npoints]
    return points

'''
# how to use these plotting fcts
hypers_prior = makeHypersPrior()
hypers = hypersSave[-1]

posteriors = make_posteriorlist(hypers)
priors = make_posteriorlist(hypers_prior)

# use same bounds and tpts if uses in one plot
# this might take long!!!
bounds = [[20,50],[0.8,1.2],[0,0.2],[0,0.4],[0,0.8],[0,2]]
tpts = 50
marginal2D_df, ts = evaluate_2D_marginals(posteriors, bounds, tpts)
marginal2D_df_prior, ts = evaluate_2D_marginals(priors, bounds, tpts)

points = extract_points(outRSave[-1], 100)


truepoint = np.array([trueK, trueX0, trueDockP, trueRibbonLambda, truerho, truescale])
plot_2d_marginals(marginal2D_df,
                  ts, cmap = 'Blues',
                 plot_prior=True, marginal2D_df_prior=marginal2D_df_prior,
                  plot_points = True, points=points, plot_hist=False,
                 plot_true=True, truepoint=truepoint)

'''

######################
def pairPlotStuff_truevalues_weights(out, wTrue, markersize=30):
    w0 = np.hstack((out[0,:], wTrue[0]))
    w1 = np.hstack((out[1,:], wTrue[1]))
    w2 = np.hstack((out[2,:], wTrue[2]))
    w3 = np.hstack((out[3,:], wTrue[3]))
    w4 = np.hstack((out[4,:], wTrue[4]))
    marker = np.zeros(len(w0))
    marker[-1] = 1
    lastindex = len(w0) -1
    
    dSetFrame = {"w0": w0, "w1": w1,"w2": w2,"w3": w3,"w4": w4, "marker":marker}
    dFrame = pd.DataFrame(dSetFrame)
    plt.figure(1)
    g = sns.PairGrid(dFrame, hue='marker', vars=["w0","w1","w2","w3","w4"], palette={0:sns.xkcd_rgb["windows blue"],1:sns.xkcd_rgb["pale red"]})
    g = g.map_upper(plt.scatter, s=markersize)
    #g = g.map_lower(plt.scatter, s=markersize)
    #g = g.map_lower(sns.kdeplot, cmap="Blues_d")
    g = g.map_lower(kde_self, cmap="Blues_d")
    #g = g.map_diag(plt.hist)#, histtype='step', linewidth=3)
    #g = g.map_diag(lambda x, **kwargs: plt.axvline(dFrame.get_value(lastindex, x.name),**kwargs))
    g = g.map_diag(hist_self)

    plt.show()

    
"""
Plot of the filters
"""

def plotFilters(t,c,a,phi, outW, wTrue,lagms, threshold=100, mean=False):
    wEst = np.mean(outW[:-1,:threshold], axis=1)

    filtT = t * .001
    filtTrue = cosFilter(t,a,c,phi,wTrue,lagms)
    filtEst = cosFilter(t,a,c,phi,wEst, lagms)
    filtMean = cosFilter(t,a,c,phi,[0.2,0.2,0.2,0.2,0.2])

    tPlot=np.arange(len(filtTrue))*0.1

    plt.figure(1, figsize=(8,6))
    plt.plot(tPlot,filtTrue, label='True')
    plt.plot(tPlot,filtEst, label='Est')
    if mean == True:
        plt.plot(tPlot,filtMean, label='Mean')

    plt.xlabel('ms')
    plt.title('Filter')
    plt.legend()
    


    

"""
NeurIPS version
"""
sns.set_style('ticks')


def plot_kernel(scales, labels, colors=[None] * 10):
    for i, scale in enumerate(scales):
        celltype = -1
        light_ca_kernel = - celltype * cone_kernel_scale(scale)
        plt.plot(light_ca_kernel, label=labels[i], color=colors[i])
        plt.legend()
        plt.xlabel('ms')
        sns.despine()
        # plt.title('light-Ca-kernel')
        plt.xticks([0, 100, 200, 300])
        plt.yticks([0])

'''
# HOW TO USE
_, _, _, _, _, scale_fit = get_expected_values(hypersSave[-1])
_, _, _, _, _, scale_prior = get_expected_values(hypersSave[0])

plt.figure(1, figsize=(4, 4))
plot_kernel([scale_fit, scale_prior], labels=['fit mean', 'prior mean'], colors=['b', 'darkorange', 'r'])

filename = 'kernelfit.svg'
# plt.savefig(filename, bbox_inches='tight', format='svg')
'''

##########################################################################



def plot_event_histo(data,  sumstatnames = ['1-q','2-q','3-q','4-q','5-q','6-q','7-q','8-q' ],
                     color='b',
                    alpha=1,
                    errcolor='black',
                    label=None):
    """
    :param data: array (ntraces, release)
    :param figsize:
    :param sumstatnames:
    :param color:
    :param alpha:
    :param errcolor:
    :return:
    """

    dn = np.zeros((4,8))
    for i in range(4):
        #dn[i,0]= np.sum(data[i])
        for j in range(0,8):
            dn[i,j]= np.sum(data[i]==j+1)

    df = pd.DataFrame(dn)

    df.columns = sumstatnames


    sns.set_context("notebook")
    sns.set_style("white")


    #plt.figure(1, figsize=(10,6))
    ax = sns.barplot( data=df,  ci='sd',capsize=.2, errwidth=2, estimator=np.mean,
                     color=color,
                     alpha=alpha,
                     errcolor=errcolor,
                    label=label)
    plt.xlabel('event type')
    plt.ylabel('count')
    plt.legend()
    sns.despine()


def plot_pairwise_loss_comparison(data, fit,w, colordata='r', colorfit='b', alpha=0.5):
    g = scp.signal.gaussian(10, 2)

    fitloss = pairwise_loss(data, g, w, mode='fit', fit=fit)
    dataloss = pairwise_loss(data, g, w, mode='data')

    # data
    df = pd.DataFrame(dataloss)
    df.columns = [None]
    ax = sns.barplot(data=df, ci='sd', capsize=.2, errwidth=2, estimator=np.mean,
                     color=colordata,
                     alpha=alpha,
                     errcolor='r',
                     label='data')

    # fit
    df = pd.DataFrame(fitloss)
    df.columns = [None]
    ax = sns.barplot(data=df, ci='sd', capsize=.2, errwidth=2, estimator=np.mean,
                     color=colorfit,
                     alpha=alpha,
                     errcolor='b',
                     label='fit')

    sns.despine()
    plt.xlabel('discrepancy')


def plot_event_loss_histo_comprison(data, fit,w, alpha=0.5, label='data', color='r', errcolor='r'):
    plt.figure(1, figsize=(12, 4))

    gs = gridspec.GridSpec(1, 2, width_ratios=[8, 1])
    ax0 = plt.subplot(gs[0])
    # plt.subplot(121)
    plot_event_histo(data, alpha=0.5, label='data', color='r', errcolor='r')
    plot_event_histo(fit, alpha=0.5, label='fit', color='b', errcolor='b')
    ax1 = plt.subplot(gs[1])
    plot_pairwise_loss_comparison(data, fit,w)

# plot_event_loss_histo_comprison(data,fit)

"""
new version: "pairwise loss"
rest vs 1


TAKE THIS AS BEST APPROXIMATION TO MEAN
"""


def pairwise_loss(data, g, w, mode, fit=None):
    """
    calculates the all-1 vs one
    if mode=data: between data traces ( mean of all-1 vs one)
    if mode=fit: between data and fit (mean data vs one fit)
    """
    w = np.copy(w)
    w[0] = 0.062
    w[-1] = 0
    nsumstats = 9
    if mode == 'data':
        fitSS = np.zeros((len(data), nsumstats))
        dists = np.zeros(len(data))
        trueG = np.zeros((len(data) - 1, len(data[0]) + len(g) - 1))

        for i in range(len(data)):
            a = list(range(len(data)))
            a.pop(i)

            count = 0
            for j in a:  # all data except one (i-th)
                trueG[count] = scipy.signal.convolve(data[j], g)
                count += 1
            dataSS = makeSS(data[np.arange(len(data)) != i], g, trueG, w)
            # print(dataSS)
            fitSS[i] = makeSS(data[i], g, trueG, w)
            # print(fitSS[i])
            # print()
            dists[i] = scipy.spatial.distance.euclidean(fitSS[i], dataSS)



    elif mode == 'fit':
        fitSS = np.zeros((len(fit), nsumstats))
        dists = np.zeros(len(fit))
        trueG = np.zeros((len(fit), len(data[0]) + len(g) - 1))

        for i in range(len(fit)):
            trueG[i, :] = scipy.signal.convolve(data[i], g)
        for i in range(len(fit)):
            # trueG = scipy.signal.convolve(data,g)
            dataSS = makeSS(data, g, trueG, w)
            print(dataSS)
            fitSS[i] = makeSS(fit[i], g, trueG, w)  # take all fitted traces
            # print(fitSS[i])
            # print()
            dists[i] = scipy.spatial.distance.euclidean(fitSS[i], dataSS)

    else:
        print('this mode is not yet implemented. choose fit or data.')

    return dists
