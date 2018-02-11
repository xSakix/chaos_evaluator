import chaos_sim as chs
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, '../etf_data')
from etf_data_loader import load_all_data_from_file

sys.path.insert(0, '../buy_hold_simulator')
from result_loader import load_ranked
import bah_simulator as bah

from scipy.stats import binom, norm
import pymc3 as pm

# -----

prefix = 'lse_'

start_date = '1993-01-01'
end_date = '2018-02-09'

df_adj_close = load_all_data_from_file(prefix + 'etf_data_adj_close.csv', start_date, end_date)


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
    sim = chs.ChaosSim(etf)
    sim.invest(df_adj_close[etf])
    sim.investor.compute_means()

    # if sim.investor.m == 0.:
    #
    #     _, ax = plt.subplots(3, 1)
    #     for rms in sim.investor.rms_list:
    #         ax[2].plot(rms)
    #
    #     print('invested:' + str(sim.investor.invested_history[-1]))
    #     print('value gained:' + str(sim.investor.history[-1]))
    #     print('ror:' + str(sim.investor.ror_history[-1]))
    #     print('mean:' + str(sim.investor.m))
    #     print('std:' + str(sim.investor.std))
    #     for rms in sim.investor.means:
    #         print(str(rms))
    #
    #     ax[0].plot(np.log(df_adj_close[etf]))
    #     ax[0].plot(np.log(sim.investor.invested_history))
    #     ax[0].plot(np.log(sim.investor.history))
    #     ax[1].plot(sim.investor.ror_history)
    #     ax[0].legend(['nav', 'invested', 'value'])
    #     ax[1].legend(['RoR'])
    #
    #     plt.show()
    return sim.investor


def compute_bah(etf):
    dca = bah.DCA(30, 300.)
    investor = bah.Investor(etf, [1.0], dca)
    sim = bah.BuyAndHoldInvestmentStrategy(investor, 2.)
    sim.invest(df_adj_close[etf])
    investor.compute_means()
    year = 1
    legends = []
    for rms in investor.rms_list:
        if len(rms) == 1:
            year += 1
            continue
        plt.plot(rms)
        legends.append('MA of returns for %d years window' % (year))
        year += 1
    plt.legend(legends)
    plt.title('Moving averages of returns')
    plt.savefig('results/ma_' + etf[0] + '.png')
    plt.clf()

    print('invested:' + str(investor.invested_history[-1]))
    print('value gained:' + str(investor.history[-1]))
    print('returns:' + str(investor.ror_history[-1]))
    print('mean:' + str(investor.m))
    print('std:' + str(investor.std))
    for rms in investor.means:
        print(str(rms))

    plt.plot(np.log(df_adj_close[etf]))
    plt.plot(np.log(sim.investor.invested_history))
    plt.plot(np.log(sim.investor.history))
    plt.title('Buy & hold value results')
    plt.legend(['nav', 'invested', 'value'])
    plt.savefig('results/bah_' + etf[0] + '.png')
    plt.clf()

    plt.plot(sim.investor.ror_history)
    plt.title('Overall returns of simulation')
    plt.legend(['returns'])
    plt.savefig('results/returns_' + etf[0] + '.png')
    plt.clf()

    return sim.investor


np.warnings.filterwarnings('ignore')

data = load_ranked(prefix)

MIN_ETFS = 0
MAX_ETFS = 1
MAX_RUNS = 10

for i in range(MIN_ETFS, MAX_ETFS):
    top = data['ticket'].iloc[i]
    print(top)

    investor = compute_bah([top])

    result_md = 'results/' + top + '.md'
    if os.path.isfile(result_md):
        os.remove(result_md)
    with open(result_md, 'a+') as fd:
        fd.write('# ' + top + '\n')
        fd.write('## BUY & HOLD results\n')
        fd.write('* Invested:' + str(investor.invested_history[-1]) + '\n')
        fd.write('* Value gained:' + str(investor.history[-1]) + '\n')
        fd.write('* Overall returns:' + str(investor.ror_history[-1]) + '\n')
        fd.write('* Mean returns:' + str(investor.m) + '\n')
        fd.write('* Std of returns:' + str(investor.std) + '\n')
        fd.write('![alt text](bah_' + top + '.png)' + '\n')
        fd.write('![alt text](returns_' + top + '.png)' + '\n')
        fd.write('![alt text](ma_' + top + '.png)' + '\n')

    investors = []
    while len(investors) < MAX_RUNS:
        investor = compute_one_etf([top])
        if investor.cash == investor.invested or investor.m == 0.:
            continue
        investors.append(investor)
        print('%d:%f:%f' % (len(investors), investor.history[-1], investor.m))

    means = []
    devs = []
    for investor in investors:
        means.append(investor.m)
        devs.append(investor.std)

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

    with open(result_md, 'a+') as fd:
        fd.write('## Chaos simulation results using Bayesian analysis' + '\n')
        fd.write('* Validity:' + str(priors[-1]) + '\n')

    grid = np.linspace(0., 1., MAX_RUNS)
    likehood = binom.pmf(np.count_nonzero(result < 0.05), len(means), grid)
    posterior = likehood * np.array(priors)
    posterior = posterior / posterior.sum()

    samples = np.random.choice(grid, p=posterior, size=int(1e4), replace=True)
    with open(result_md, 'a+') as fd:
        fd.write('* Maximum posteriori at prob =(%f,%f)' % (max(posterior), grid[posterior == max(posterior)]) + '\n')
        fd.write('* Migh posterior density percentile interval 95:' + str(pm.hpd(samples, alpha=0.95)) + '\n')

    print('maximum posteriori at prob =(%f,%f)' % (max(posterior), grid[posterior == max(posterior)]))
    print('high posterior density percentile interval 95:' + str(pm.hpd(samples, alpha=0.95)))

    _, (ax0, ax1, ax2) = plt.subplots(1, 3)
    ax0.plot(grid, posterior)
    ax0.set_title('probability distribution of mean returns')
    ax1.plot(samples, 'o')
    ax1.set_title('samples of mean returns')
    sns.kdeplot(samples, ax=ax2)
    ax2.set_title('probability distribution of mean returns from samples')
    plt.savefig('results/distribution1_' + top + '.png')
    plt.show()

    with open(result_md, 'a+') as fd:
        fd.write('![alt text](distribution1_' + top + '.png)' + '\n')

    with pm.Model() as model:
        mu = pm.Normal('mu', mu=np.mean(means), sd=np.std(means))
        sigma = pm.Uniform('sigma', lower=min(devs), upper=max(devs))
        mean_returns = pm.Normal('mean_returns', mu=mu, sd=sigma, observed=np.array(means))
        mean_q = pm.find_MAP()
        std_mu = ((1 / pm.find_hessian(mean_q, vars=[mu])) ** 0.5)[0]
        std_sigma = ((1 / pm.find_hessian(mean_q, vars=[sigma])) ** 0.5)[0]

    samples = norm.rvs(loc=mean_q['mu'], scale=mean_q['sigma'], size=10000)
    print('89 percentile:' + str(pm.hpd(samples, alpha=0.89)))
    print('95 percentile:' + str(pm.hpd(samples, alpha=0.95)))
    with open(result_md, 'a+') as fd:
        fd.write('* 89 percentile:' + str(pm.hpd(samples, alpha=0.89)) + '\n')
        fd.write('* 95 percentile:' + str(pm.hpd(samples, alpha=0.95)) + '\n')

    _, (ax0, ax1) = plt.subplots(1, 2)
    ax0.plot(means)
    ax0.set_title('Means of returns gained from Chaos simulation')
    sns.kdeplot(samples, ax=ax1)
    ax1.set_title('Distrinution of mean returns from Chaos simulation')
    plt.savefig('results/distribution2_' + top + '.png')
    plt.show()
    with open(result_md, 'a+') as fd:
        fd.write('![alt text](distribution2_' + top + '.png)' + '\n')
