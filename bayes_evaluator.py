import chaos_sim as chaos
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timeit

sys.path.insert(0, '../etf_data')
from etf_data_loader import load_all_data_from_file

import pymc3 as pm
import seaborn as sns
import random
from datetime import datetime


def gen_random_date(year_low, year_high):
    y = random.randint(year_low, year_high)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return datetime(year=y, month=m, day=d)


def get_data_random_dates(df_adj_close):
    rand_start = gen_random_date(2010, 2017)
    rand_end = gen_random_date(2010, 2017)
    if rand_start > rand_end:
        tmp = rand_start
        rand_start = rand_end
        rand_end = tmp

    print(rand_start)
    print(rand_end)
    data = df_adj_close[df_adj_close['Date'] > str(rand_start)]
    data = data[data['Date'] < str(rand_end)]

    return data


def compute_one_etf(etf, data):
    sim = chaos.ChaosSim(etf, only_buy=True)
    sim.invest(data)
    # _, ax = plt.subplots(3, 1)
    #
    # ax[0].plot(data)
    # ax[1].plot(sim.investor.invested_history)
    # ax[1].plot(sim.investor.history)
    # ax[2].plot(sim.investor.ror_history)
    # ax[0].legend(['nav'])
    # ax[1].legend(['invested', 'value'])
    # ax[2].legend(['Returns'])
    #
    # plt.show()

    return sim.investor


print('starting to load data')
prefix = ''
start_date = '1993-01-01'
end_date = '2017-12-31'
df_adj_close = load_all_data_from_file(prefix + 'etf_data_adj_close.csv', start_date, end_date)

etf = ['SPY']

data = df_adj_close[etf].values
print(len(data))

ror = []

while len(ror) < 30:
    # df_price = get_data_random_dates(df_adj_close)
    investor = compute_one_etf(etf, df_adj_close[etf])
    if investor.invested == investor.history[-1]:
        continue
    print('invested:' + str(investor.invested_history[-1]))
    print('value gained:' + str(investor.history[-1]))
    print('ror:' + str(investor.ror_history[-1]))
    ror.append(np.array(investor.ror_history))

data_list = []
for d in data:
    if len(d) > 0 and d[0] != 0 and not np.isnan(d[0]):
        data_list.append(d[0])
    elif d > 0 and not np.isnan(d):
        data_list.append(d)

data = np.array(data_list)

mean_data = data.mean()
std_data = data.std()

data_s = (data - mean_data) / std_data
# data_s = data
data_s2 = data_s ** 2
data_s3 = data_s ** 3
data_s4 = data_s ** 4

print('mean of prices: ' + str(mean_data))
print('std of prices: ' + str(std_data))

for r in ror:
    plt.plot(data_s, r)
plt.title('nav to returns')
plt.xlabel('nav')
plt.ylabel('returns')
plt.show()

sns.kdeplot(data)
plt.title('distribution of nav')
plt.show()

for r in ror:
    sns.kdeplot(r)
plt.title('distribution of returns')
plt.show()

df = pd.DataFrame({'data_s': data_s, 'ror': ror[0]})
print(df.corr().round(2))

with pm.Model() as model:
    alpha = pm.Normal(name='alpha', mu=mean_data, sd=std_data)
    beta = pm.Normal(name='beta', mu=0, sd=10, shape=4)
    sigma = pm.Uniform(name='sigma', lower=0, upper=std_data)
    mu = pm.Deterministic('mu', alpha + beta[0] * data_s + beta[1] * data_s2 + beta[2] * data_s3 + beta[3] * data_s4)
    ret = pm.Normal(name='returns', mu=mu, sd=sigma, observed=ror)
    trace_model = pm.sample(1000, tune=2000)

print(pm.summary(trace_model, ['alpha', 'beta', 'sigma']))

pm.traceplot(trace_model, varnames=['alpha', 'beta', 'sigma'])
plt.title('model parameters')
plt.show()

mu_pred = trace_model['mu']
idx = np.argsort(data_s)
mu_hpd = pm.hpd(mu_pred, alpha=0.11)[idx]
ret_pred = pm.sample_ppc(trace_model, 10000, model)
ret_pred_hpd = pm.hpd(ret_pred['returns'], alpha=0.11)[idx]

print('89 percentile: %f +/- %f' % (np.mean(ret_pred_hpd), np.std(ret_pred_hpd)))

for r in ror:
    plt.plot(r)
plt.plot(ret_pred_hpd)
plt.show()

for r in ror:
    # plt.scatter(data_s, r, c='C0', alpha=0.3)
    plt.plot(data_s[idx], r, alpha=0.3)
plt.fill_between(data_s[idx], mu_hpd[:, 0], mu_hpd[:, 1], color='C2', alpha=0.25)
plt.fill_between(data_s[idx], ret_pred_hpd[:, 0], ret_pred_hpd[:, 1], color='C2', alpha=0.25)
plt.show()
