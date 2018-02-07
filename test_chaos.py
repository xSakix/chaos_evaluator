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

data = load_ranked()

top = data.ticket[0]
print(top)

investors = []
while len(investors) < 20:
    investor = compute_one_etf([top])
    if investor.cash == investor.invested:
        continue
    investors.append(investor)
    print(len(investors))

means = []
for investor in investors:
    means.append(investor.m)


print('original:'+str(data.mean[0]))
print('observed:'+str(np.mean(means)))

#plt.plot(means)
sns.kdeplot(means)
plt.show()

