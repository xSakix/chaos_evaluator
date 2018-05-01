import numpy as np
import sys

sys.path.insert(0, '../buy_hold_simulation')

from bah_simulator import Investor


def yorke(x, r):
    return x * r * (1. - x)


class ChaosSim:
    def __init__(self, ticket, dist=[1.], tr_cost=2.,only_buy=False, sum=300,period=30):
        self.investor = Investor(ticket,dist)
        self.investor.dca.cash=sum
        self.investor.dca.period = period
        self.tr_cost = tr_cost
        self.r = np.random.uniform(2.9, 3.9,2)
        self.only_buy = only_buy

    def invest(self, data):

        if len(data.keys()) == 0:
            return

        x = np.array([0.01, 0.01])
        for _ in range(20):
            x = yorke(x, self.r)

        self.investor.shares = np.zeros(len(data.keys()))

        day = 0

        for i in data.index:
            prices = data.loc[i].values
            if prices == 0.:
                continue
            portfolio = self.investor.cash + np.dot(prices, self.investor.shares)
            if np.isnan(portfolio):
                portfolio = 0.

            self.investor.history.append(portfolio)
            self.investor.invested_history.append(self.investor.invested)

            if self.investor.invested == 0:
                ror = 0
            else:
                ror = (portfolio - self.investor.invested) / self.investor.invested
            self.investor.ror_history.append(ror)

            x = yorke(x, self.r)

            if day % self.investor.dca.period == 0:
                self.investor.cash += self.investor.dca.cash
                self.investor.invested += self.investor.dca.cash

            if x[0] >= 0.9 and sum(self.investor.shares > 0) > 0 and not self.only_buy:
                self.investor.cash = np.dot(self.investor.shares, prices) - sum(self.investor.shares > 0) * self.tr_cost
                self.investor.shares = np.zeros(len(data.keys()))

            if x[1] >= 0.9 and self.investor.cash > prices:
                portfolio = self.investor.cash + np.dot(prices, self.investor.shares)
                c = np.multiply(self.investor.dist, portfolio)
                c = np.subtract(c, self.tr_cost)
                s = np.divide(c, prices)
                s = np.floor(s)
                self.investor.shares = s
                self.investor.cash = portfolio - np.dot(self.investor.shares, prices) - len(s) * self.tr_cost

            day += 1
