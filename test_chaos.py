import chaos_sim as chs
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tabulate

sys.path.insert(0, '../etf_data')
from etf_data_loader import load_all_data_from_file

sys.path.insert(0, '../buy_hold_simulation')
from result_loader import load_ranked
import bah_simulator as bah

from scipy.stats import binom, norm
import pymc3 as pm

from md2pdf.doc import Document

from datetime import datetime
import random

# -----

prefix = ''

start_date = '1993-01-01'
end_date = '2017-12-31'

df_adj_close = load_all_data_from_file(prefix + 'etf_data_adj_close.csv', start_date, end_date)


def gen_random_date(year_low, year_high):
    y = random.randint(year_low, year_high)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return datetime(year=y, month=m, day=d)


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

    data = get_data_random_dates()

    sim.invest(data[etf])
    sim.investor.compute_means()

    return sim.investor


def get_data_random_dates():
    rand_start = gen_random_date(1993, 2017)
    rand_end = gen_random_date(1993, 2017)
    if rand_start > rand_end:
        tmp = rand_start
        rand_start = rand_end
        rand_end = tmp
    data = df_adj_close[df_adj_close['Date'] > str(rand_start)]
    data = data[data['Date'] < str(rand_end)]

    return data


def process_bah_investor(investor, etf):
    year = 1
    legends = []
    for rms in investor.rms_list:
        if len(rms) == 1:
            year += 1
            continue
        plt.plot(rms)
        legends.append('MA returns %d period' % (year * 365))
        year += 1
    plt.legend(legends, loc='lower right')
    plt.title('Moving averages of returns')
    plt.savefig('results/ma_' + etf + '.png')
    plt.clf()

    plt.plot(investor.invested_history)
    plt.plot(investor.history)
    plt.title('Buy & hold value results')
    plt.legend(['nav', 'invested', 'value'])
    plt.savefig('results/bah_' + etf + '.png')
    plt.clf()

    plt.plot(investor.ror_history)
    plt.title('Overall returns of simulation')
    plt.legend(['returns'])
    plt.savefig('results/returns_' + etf + '.png')
    plt.clf()


def compute_bah(etf):
    dca = bah.DCA(30, 300.)
    investor = bah.Investor(etf, [1.0], dca)
    sim = bah.BuyAndHoldInvestmentStrategy(investor, 2.)
    sim.invest(df_adj_close[etf])
    investor.compute_means()

    return investor


def run_bah_sim(etf):
    data = get_data_random_dates()

    dca = bah.DCA(30, 300.)
    investor = bah.Investor(etf, [1.0], dca)
    sim = bah.BuyAndHoldInvestmentStrategy(investor, 2.)
    sim.invest(data[etf])
    investor.compute_means()

    return investor


class Report:
    def __init__(self, etf, bah_investor, means_bah, devs_bah, hpd89_bah, samples_bah_mean_returns, means_chaos,
                 devs_chaos, hpd89_chaos, samples_chaos_mean_returns):
        self.bah_investor = bah_investor
        self.etf = etf

        self.means_bah = means_bah
        self.devs_bah = devs_bah
        self.hpd89_bah = hpd89_bah
        self.samples_bah_mean_returns = samples_bah_mean_returns

        self.means_chaos = means_chaos
        self.devs_chaos = devs_chaos
        self.hpd89_chaos = hpd89_chaos
        self.samples_chaos_mean_returns = samples_chaos_mean_returns

    def gen_report(self):
        process_bah_investor(self.bah_investor, self.etf)
        self.original_mean = bah_investor.m

        result_md = 'results/' + top + '.md'
        if os.path.isfile(result_md):
            os.remove(result_md)
        with open(result_md, 'a+') as fd:
            fd.write('# ' + top + '\n')
            fd.write('## 1. BUY & HOLD results\n')
            fd.write('* Invested: ' + str(np.round(self.bah_investor.invested_history[-1], 2)) + '\n')
            fd.write('* Value gained: ' + str(np.round(self.bah_investor.history[-1], 2)) + '\n')
            fd.write('* Overall returns :' + str(np.round(self.bah_investor.ror_history[-1], 2)) + '\n')
            fd.write('* Mean returns: ' + str(np.round(self.bah_investor.m, 2)) + '\n')
            fd.write('* Std of returns(volatility): ' + str(np.round(self.bah_investor.std, 4)) + '\n')
            fd.write('![alt text](bah_' + top + '.png)' + '\n')
            fd.write('![alt text](returns_' + top + '.png)' + '\n')
            fd.write('![alt text](ma_' + top + '.png)' + '\n')

        with open(result_md, 'a+') as fd:
            fd.write(
                '* Validity: ' + str(
                    np.count_nonzero(np.abs(self.means_bah - self.original_mean) < 0.05) / len(
                        self.means_bah) * 100.) + '% \n')

        with open(result_md, 'a+') as fd:
            fd.write('* 89 percentile: ' + str(np.round(np.mean(self.hpd89_bah), 2)) + '\n')

        plt.plot(self.means_bah)
        plt.title('Means of returns gained from b&h simulations')
        plt.savefig('results/distribution_bah_means_' + self.etf + '.png')
        plt.clf()

        sns.kdeplot(self.samples_bah_mean_returns[:, 0])
        sns.kdeplot(self.samples_bah_mean_returns[:, 1])
        plt.title('Distribution of mean returns from b&h simulations')
        plt.savefig('results/distribution_bah_kde_' + self.etf + '.png')
        plt.clf()

        plt.plot(self.hpd89_bah[:, 0])
        plt.plot(self.hpd89_bah[:, 1])
        plt.title('sampled means bounded by 89 percentile')
        plt.savefig('results/sampled_bah_89_' + self.etf + '.png')
        plt.legend(['mean', 'lower 89%', 'upper 89%'])
        plt.clf()

        with open(result_md, 'a+') as fd:
            fd.write('![alt text](distribution_bah_means_' + self.etf + '.png)' + '\n')
            fd.write('![alt text](distribution_bah_kde_' + self.etf + '.png)' + '\n')
            fd.write('![alt text](sampled_bah_89_' + self.etf + '.png)' + '\n')

        with open(result_md, 'a+') as fd:
            fd.write('## 2. Chaos simulation results using Bayesian analysis' + '\n')
            fd.write(
                '* Validity: ' + str(
                    np.count_nonzero(np.abs(self.means_chaos - self.original_mean) < 0.05) / len(
                        self.means_chaos) * 100.) + '% \n')

        with open(result_md, 'a+') as fd:
            fd.write('* 89 percentile: ' + str(np.round(np.mean(self.hpd89_chaos), 2)) + '\n')

        plt.plot(self.means_chaos)
        plt.title('Means of returns gained from Chaos simulation')
        plt.savefig('results/distribution2_means_' + self.etf + '.png')
        plt.clf()

        sns.kdeplot(self.samples_chaos_mean_returns[:, 0])
        sns.kdeplot(self.samples_chaos_mean_returns[:, 1])
        plt.title('Distribution of mean returns from Chaos simulation')
        plt.savefig('results/distribution2_kde_' + self.etf + '.png')
        plt.clf()

        plt.plot(self.hpd89_chaos[:, 0])
        plt.plot(self.hpd89_chaos[:, 1])
        plt.title('sampled means bounded by 89 percentile')
        plt.savefig('results/sampled_89_' + self.etf + '.png')
        plt.legend(['mean', 'lower 89%', 'upper 89%'])
        plt.clf()

        with open(result_md, 'a+') as fd:
            fd.write('![alt text](distribution2_means_' + self.etf + '.png)' + '\n')
            fd.write('![alt text](distribution2_kde_' + self.etf + '.png)' + '\n')
            fd.write('![alt text](sampled_89_' + self.etf + '.png)' + '\n')

        # generate report
        doc = Document.from_markdown(result_md)
        doc.save_to_pdf(pdf_file_name='results/' + self.etf + '.pdf')
        os.remove(result_md)

        # clean up
        for f in os.listdir('results/'):
            if f.endswith('.png'):
                os.remove('results/' + f)


np.warnings.filterwarnings('ignore')

data = load_ranked(prefix)

MIN_ETFS = 51
MAX_ETFS = 52
MAX_RUNS = 100

result_csv = 'results/result.csv'
if os.path.isfile(result_csv):
    result_df = pd.read_csv(result_csv)
else:
    result_df = pd.DataFrame(
        columns=['ticket', 'original_mean', 'original_std', 'hpd89_bah', 'hpd89_chaos', 'validity_bah', 'validity_chaos'])

for i in range(MIN_ETFS, MAX_ETFS):
    top = data['ticket'].iloc[i]
    # top = 'XLK'
    print(top)

    bah_investor = compute_bah([top])
    original_mean = bah_investor.m
    original_std = bah_investor.std
    print('invested:' + str(bah_investor.invested_history[-1]))
    print('value gained:' + str(bah_investor.history[-1]))
    print('returns:' + str(bah_investor.ror_history[-1]))
    print('mean:' + str(bah_investor.m))
    print('std:' + str(bah_investor.std))
    for rms in bah_investor.means:
        print(str(rms))

    investors = []
    while len(investors) < MAX_RUNS:
        investor = run_bah_sim([top])
        if investor.m == 0.:
            continue
        investors.append(investor)
        print('%d:%f:%f' % (len(investors), investor.history[-1], investor.m))

    means_bah = [investor.m for investor in investors]
    devs_bah = [investor.std for investor in investors]

    means_bah = np.array(means_bah)
    devs_bah = np.array(devs_bah)
    print('original:%f +/- %f' % (original_mean, original_std))
    print('observed:%f +/- %f' % (np.mean(means_bah), np.std(means_bah)))

    with pm.Model() as model:
        mu = pm.Normal('mu', mu=np.mean(means_bah), sd=np.std(means_bah))
        sigma = pm.Uniform('sigma', lower=np.min(devs_bah), upper=np.max(devs_bah))
        mean_returns = pm.Normal('mean_returns', mu=mu, sd=sigma, observed=np.array(means_bah))
        trace_model = pm.sample(1000, tune=1000)

    samples_bah = pm.sample_ppc(trace_model, size=100, model=model)

    hpd89_bah = pm.hpd(samples_bah['mean_returns'], alpha=0.89)

    print('mean 89 percentile:' + str(np.mean(hpd89_bah)))

    investors = []
    while len(investors) < MAX_RUNS:
        investor = compute_one_etf([top])
        if investor.cash == investor.invested or investor.m == 0.:
            continue
        investors.append(investor)
        print('%d:%f:%f' % (len(investors), investor.history[-1], investor.m))

    means = [investor.m for investor in investors]
    devs = [investor.std for investor in investors]

    means = np.array(means)
    devs = np.array(devs)
    print('original:%f +/- %f' % (original_mean, original_std))
    print('observed:%f +/- %f' % (np.mean(means), np.std(devs)))

    with pm.Model() as model:
        mu = pm.Normal('mu', mu=np.mean(means), sd=np.std(devs))
        sigma = pm.Uniform('sigma', lower=np.min(devs), upper=np.max(devs))
        mean_returns = pm.Normal('mean_returns', mu=mu, sd=sigma, observed=np.array(means))
        trace_model = pm.sample(1000, tune=1000)

    samples_chaos = pm.sample_ppc(trace_model, size=100, model=model)

    hpd89_chaos = pm.hpd(samples_chaos['mean_returns'], alpha=0.89)

    print('mean 89 percentile:' + str(np.mean(hpd89_chaos)))

    validity_chaos = np.count_nonzero(np.abs(means - original_mean) < 0.05) / len(means) * 100.
    validity_bah = np.count_nonzero(np.abs(means_bah - original_mean) < 0.05) / len(means_bah) * 100.

    result_df = result_df.append({
        'ticket': top,
        'original_mean': original_mean,
        'original_std': original_std,
        'hpd89_bah': np.mean(hpd89_bah),
        'hpd89_chaos': np.mean(hpd89_chaos),
        'validity_bah': validity_bah,
        'validity_chaos': validity_chaos}, ignore_index=True)

    report = Report(top, bah_investor, means_bah, devs_bah, hpd89_bah, samples_bah['mean_returns'], means, devs,
                    hpd89_chaos, samples_chaos['mean_returns'])
    report.gen_report()

result_df.to_csv(result_csv)
