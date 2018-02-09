import chaos_sim as chs
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timeit
import seaborn as sns

sys.path.insert(0, '../etf_data')
from etf_data_loader import load_all_data_from_file

sys.path.insert(0, '../buy_hold_simulator')
from result_loader import load_ranked

from scipy.stats import binom
import pymc3 as pm


def rolling_mean(data, period):
    rm = pd.rolling_mean(data, period)
    rm = rm[~np.isnan(rm)]
    return rm


def mean(value):
    value = np.mean(value)
    if np.isnan(value):
        return 0.
    return value


def compute_one_etf(etf):
    start_date = '1993-01-01'
    end_date = '2017-12-31'

    df_adj_close = load_all_data_from_file('etf_data_adj_close.csv', start_date, end_date)
    sim = chs.ChaosSim(etf)
    sim.invest(df_adj_close[etf])
    sim.investor.compute_means()

    # _, ax = plt.subplots(3, 1)
    # for rms in sim.investor.rms_list:
    #     ax[2].plot(rms)
    #
    # print('invested:' + str(sim.investor.invested_history[-1]))
    # print('value gained:' + str(sim.investor.history[-1]))
    # print('ror:' + str(sim.investor.ror_history[-1]))
    # print('mean:' + str(sim.investor.m))
    # print('std:' + str(sim.investor.std))
    # for rms in sim.investor.means:
    #     print(str(rms))
    #
    # ax[0].plot(np.log(df_adj_close[etf]))
    # ax[0].plot(np.log(sim.investor.invested_history))
    # ax[0].plot(np.log(sim.investor.history))
    # ax[1].plot(sim.investor.ror_history)
    # ax[0].legend(['nav', 'invested', 'value'])
    # ax[1].legend(['RoR'])
    #
    # plt.show()
    return sim.investor


np.warnings.filterwarnings('ignore')

data = load_ranked()

MAX_ETFS = 1
MAX_RUNS = 100

for i in range(0, MAX_ETFS):
    top = data['ticket'].iloc[i]
    print(top)

    investors = []
    while len(investors) < MAX_RUNS:
        investor = compute_one_etf([top])
        if investor.cash == investor.invested:
            continue
        investors.append(investor)
        # print('%d:%f:%f' % (len(investors), investor.history[-1],investor.m))

    means = []
    for investor in investors:
        means.append(investor.m)

    original_mean = data.iloc[i]['mean']
    original_std = data.iloc[i]['std']
    print('original:%f +/- %f' % (original_mean, original_std))
    print('observed:%f +/- %f' % (np.mean(means), np.std(means)))

    priors = []
    means = np.array(means)
    for i in range(1, len(means) + 1):
        means_sub = means[:i]
        result = np.abs(means_sub - original_mean)
        prior = np.count_nonzero(result < 0.05) / len(means_sub)
        priors.append(prior)

    print('validity:' + str(priors[-1]))

    grid = np.linspace(0., 1., MAX_RUNS)
    likehood = binom.pmf(np.count_nonzero(result < 0.05), len(means), grid)
    posterior = likehood * np.array(priors)
    posterior = posterior / posterior.sum()

    samples = np.random.choice(grid, p=posterior, size=int(1e4), replace=True)
    print('maximum posteriori at prob =(%f,%f)' % (max(posterior), grid[posterior == max(posterior)]))
    print('high posterior density percentile interval 95:' + str(pm.hpd(samples, alpha=0.95)))

    # _, (ax0, ax1, ax2) = plt.subplots(1, 3)
    # ax0.plot(grid, posterior)
    # ax1.plot(samples, 'o')
    # sns.kdeplot(samples, ax=ax2)
    # plt.show()

    # plt.plot(means)
    # sns.kdeplot(means)
    # plt.show()
