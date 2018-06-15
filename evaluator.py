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

prefix = 'mil_'

start_date = '1993-01-01'
end_date = '2017-12-31'
# end_date = '2017-12-31'

df_adj_close = load_all_data_from_file(prefix + 'etf_data_adj_close.csv', start_date, end_date)
period = 365
cash_sum = 300.


def gen_random_date(year_low, year_high):
    y = random.randint(year_low, year_high)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return datetime(year=y, month=m, day=d)


def compute_one_etf(etf,period,cash_sum):
    sim = chs.ChaosSim(etf, only_buy=True, sum=cash_sum, period=period)
    data = get_data_random_dates()
    sim.invest(data[etf])

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
    plt.plot(investor.invested_history)
    plt.plot(investor.history)
    plt.title('Buy & hold value results')
    plt.legend(['invested', 'value'])
    plt.savefig('results/bah_' + etf + '.png')
    plt.clf()

    plt.plot(investor.ror_history)
    plt.title('Overall returns of simulation')
    plt.legend(['returns'])
    plt.savefig('results/returns_' + etf + '.png')
    plt.clf()


def compute_bah(etf,period,cash_sum):
    dca = bah.DCA(period, cash_sum)
    investor = bah.Investor(etf, [1.0], dca)
    sim = bah.BuyAndHoldInvestmentStrategy(investor, 2.)
    sim.invest(df_adj_close[etf])

    return investor


def run_bah_sim(etf,period,cash_sum):
    data = get_data_random_dates()

    dca = bah.DCA(period, cash_sum)
    investor = bah.Investor(etf, [1.0], dca)
    sim = bah.BuyAndHoldInvestmentStrategy(investor, 2.)
    sim.invest(data[etf])

    return investor


class Report:
    def __init__(self,
                 etf,
                 bah_investor,
                 returns_bah,
                 hpd89_bah,
                 samples_bah_returns,
                 returns_chaos,
                 hpd89_chaos,
                 samples_chaos_returns):
        self.bah_investor = bah_investor
        self.etf = etf

        self.returns_bah = returns_bah
        self.hpd89_bah = hpd89_bah
        self.samples_bah_returns = samples_bah_returns

        self.means_chaos = returns_chaos
        self.hpd89_chaos = hpd89_chaos
        self.samples_chaos_returns = samples_chaos_returns

    def gen_report(self):
        process_bah_investor(self.bah_investor, self.etf)
        self.original_return = self.bah_investor.ror_history[-1]

        result_md = 'results/' + top + '.md'
        if os.path.isfile(result_md):
            os.remove(result_md)
        with open(result_md, 'a+') as fd:
            fd.write('# ' + top + '\n')
            fd.write('## 1. BUY & HOLD results\n')
            fd.write('* Invested: ' + str(np.round(self.bah_investor.invested_history[-1], 2)) + '\n')
            fd.write('* Value gained: ' + str(np.round(self.bah_investor.history[-1], 2)) + '\n')
            fd.write('* Overall returns :' + str(np.round(self.bah_investor.ror_history[-1], 2)) + '\n')
            fd.write('![alt text](bah_' + top + '.png)' + '\n')
            fd.write('![alt text](returns_' + top + '.png)' + '\n')

        with open(result_md, 'a+') as fd:
            fd.write(
                '* Validity: ' + str(
                    np.count_nonzero(np.abs(self.returns_bah - self.original_return) < 0.05) / len(
                        self.returns_bah) * 100.) + '% \n')

        with open(result_md, 'a+') as fd:
            fd.write('* 89 percentile: ' + str(np.round(np.mean(self.hpd89_bah), 2)) + '\n')

        plt.plot(self.returns_bah)
        plt.title('Returns gained from b&h simulations')
        plt.savefig('results/distribution_bah_means_' + self.etf + '.png')
        plt.clf()

        sns.kdeplot(self.samples_bah_returns[:, 0])
        sns.kdeplot(self.samples_bah_returns[:, 1])
        plt.title('Distribution of returns from b&h simulations')
        plt.savefig('results/distribution_bah_kde_' + self.etf + '.png')
        plt.clf()

        plt.plot(self.hpd89_bah[:, 0])
        plt.plot(self.hpd89_bah[:, 1])
        plt.title('sampled returns bounded by 89 percentile')
        plt.savefig('results/sampled_bah_89_' + self.etf + '.png')
        plt.legend(['returns', 'lower 89%', 'upper 89%'])
        plt.clf()

        with open(result_md, 'a+') as fd:
            fd.write('![alt text](distribution_bah_means_' + self.etf + '.png)' + '\n')
            fd.write('![alt text](distribution_bah_kde_' + self.etf + '.png)' + '\n')
            fd.write('![alt text](sampled_bah_89_' + self.etf + '.png)' + '\n')

        with open(result_md, 'a+') as fd:
            fd.write('## 2. Chaos simulation results using Bayesian analysis' + '\n')
            fd.write(
                '* Validity: ' + str(
                    np.count_nonzero(np.abs(self.means_chaos - self.original_return) < 0.05) / len(
                        self.means_chaos) * 100.) + '% \n')

        with open(result_md, 'a+') as fd:
            fd.write('* 89 percentile: ' + str(np.round(np.mean(self.hpd89_chaos), 2)) + '\n')

        plt.plot(self.means_chaos)
        plt.title('Returns gained from Chaos simulation')
        plt.savefig('results/distribution2_means_' + self.etf + '.png')
        plt.clf()

        sns.kdeplot(self.samples_chaos_returns[:, 0])
        sns.kdeplot(self.samples_chaos_returns[:, 1])
        plt.title('Distribution of returns from Chaos simulation')
        plt.savefig('results/distribution2_kde_' + self.etf + '.png')
        plt.clf()

        plt.plot(self.hpd89_chaos[:, 0])
        plt.plot(self.hpd89_chaos[:, 1])
        plt.title('sampled returns bounded by 89 percentile')
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

MIN_ETFS = 5
MAX_ETFS = 55
MAX_RUNS = 1000

result_csv = 'results/result_spy.csv'
if os.path.isfile(result_csv):
    result_df = pd.read_csv(result_csv)
else:
    result_df = pd.DataFrame(
        columns=['ticket', 'original_returns', 'hpd89_bah', 'hpd89_chaos', 'validity_bah',
                 'validity_chaos'])


def process_one_etf(top, result_df):
    print(top)
    bah_investor = compute_bah([top],period,cash_sum)
    print('invested:' + str(bah_investor.invested_history[-1]))
    print('value gained:' + str(bah_investor.history[-1]))
    print('returns:' + str(bah_investor.ror_history[-1]))
    investors = []
    while len(investors) < MAX_RUNS:
        investor = run_bah_sim([top],period,cash_sum)
        if len(investor.ror_history) == 0:
            continue
        investors.append(investor)
        print('%d:%f:%f:%f' % (len(investors), investor.invested, investor.history[-1], investor.ror_history[-1]))
    returns_bah = [investor.ror_history[-1] for investor in investors]
    returns_bah = np.array(returns_bah)
    # returns_bah = np.sort(returns_bah)
    print('original:%f' % bah_investor.ror_history[-1])
    print('observed:%f +/- %f' % (np.mean(returns_bah), np.std(returns_bah)))
    with pm.Model() as model:
        mu = pm.Normal('mu', mu=np.mean(returns_bah), sd=np.std(returns_bah))
        sigma = pm.Uniform('sigma', lower=0., upper=np.std(returns_bah))
        mean_returns = pm.Normal('mean_returns', mu=mu, sd=sigma, observed=np.array(returns_bah))
        trace_model = pm.sample(1000, tune=2000)
    samples_bah = pm.sample_ppc(trace_model, size=10000, model=model)
    hpd89_bah = pm.hpd(samples_bah['mean_returns'], alpha=0.11)
    print('mean 89 percentile:' + str(np.mean(hpd89_bah)))
    investors = []
    while len(investors) < MAX_RUNS:
        investor = compute_one_etf([top],period,cash_sum)
        if investor.cash == investor.invested:
            continue
        if len(investor.ror_history) == 0:
            continue
        investors.append(investor)
        print('%d:%f:%f:%f' % (len(investors), investor.invested, investor.history[-1], investor.ror_history[-1]))
    returns_chaos = [investor.ror_history[-1] for investor in investors]
    returns_chaos = np.array(returns_chaos)
    # returns_chaos = np.sort(returns_chaos)
    print('original:%f' % (bah_investor.ror_history[-1]))
    print('observed:%f +/- %f' % (np.mean(returns_chaos), np.std(returns_chaos)))
    with pm.Model() as model:
        mu = pm.Normal('mu', mu=np.mean(returns_chaos), sd=np.std(returns_chaos))
        sigma = pm.Uniform('sigma', lower=0., upper=np.std(returns_chaos))
        mean_returns = pm.Normal('mean_returns', mu=mu, sd=sigma, observed=np.array(returns_chaos))
        trace_model = pm.sample(1000, tune=2000)
    samples_chaos = pm.sample_ppc(trace_model, size=10000, model=model)
    hpd89_chaos = pm.hpd(samples_chaos['mean_returns'], alpha=0.11)
    print('mean 89 percentile:' + str(np.mean(hpd89_chaos)))
    validity_chaos = np.count_nonzero(np.abs(returns_chaos - bah_investor.ror_history[-1]) < 0.05) / len(
        returns_chaos) * 100.
    validity_bah = np.count_nonzero(np.abs(returns_bah - bah_investor.ror_history[-1]) < 0.05) / len(returns_bah) * 100.
    result_df = result_df.append({
        'ticket': top,
        'original_returns': bah_investor.ror_history[-1],
        'hpd89_bah': np.mean(hpd89_bah),
        'hpd89_chaos': np.mean(hpd89_chaos),
        'validity_bah': validity_bah,
        'validity_chaos': validity_chaos}, ignore_index=True)
    report = Report(top,
                    bah_investor,
                    returns_bah,
                    hpd89_bah,
                    samples_bah['mean_returns'],
                    returns_chaos,
                    hpd89_chaos,
                    samples_chaos['mean_returns'])
    report.gen_report()
    result_df.to_csv(result_csv, index=False)
    return result_df


# for i in range(MIN_ETFS, MAX_ETFS):
for i in range(0, 10):
    # top = data['ticket'].iloc[i]
    top = 'SPY'
    try:
        result_df = process_one_etf(top, result_df)
    except Exception as e:
        print('Error processing:' + top)
        print(e)
