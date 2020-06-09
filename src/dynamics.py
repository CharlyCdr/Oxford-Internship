import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import matplotlib as mpl
from scipy.signal import find_peaks

import graphics_class as gc


class Dynamics(object):

    def __init__(self, e, t_max, store=None):
        self.eco = e
        self.t_max = t_max
        self.n = self.eco.n
        self.prices = np.zeros((t_max + 1, self.n))
        self.wages = np.zeros(t_max + 1)
        self.prices_net = np.zeros((t_max + 1, self.n))
        self.prods = np.zeros((t_max + 1, self.n))
        self.targets = np.zeros((t_max + 1, self.n))
        self.stocks = np.zeros((t_max + 1, self.n))
        self.profits = np.zeros((t_max + 1, self.n))
        self.balance = np.zeros((t_max + 1, self.n + 1))
        self.cashflow = np.zeros((t_max + 1, self.n))
        self.tradeflow = np.zeros((t_max + 1, self.n + 1))
        self.supply = np.zeros((t_max + 1, self.n + 1))
        self.demand = np.zeros((t_max + 1, self.n + 1))
        self.tradereal = np.zeros((t_max + 1, self.n + 1))
        self.s_vs_d = np.zeros((t_max + 1, self.n + 1))
        self.b_vs_c = np.zeros(t_max + 1)
        self.Q_real = np.zeros((t_max + 1, self.n + 1, self.n + 1))
        self.Q_demand = np.zeros((t_max + 1, self.n + 1, self.n + 1))
        self.mu = np.zeros(t_max + 1)
        self.budget = np.zeros(t_max + 1)
        self.budget_res = np.zeros(t_max + 1)
        self.utility = np.zeros(t_max + 1)
        self.labour = np.zeros(t_max + 1)

        self.store = store
        self.dftemp = None
        if self.store:
            stored_funda = ['Wage/Prices', 'Labour/Productions', 'Budget/Stocks', 'Mu/Targets', 'Cons', 'Demand']
            first_index = np.array([[str(t) for k in range(len(stored_funda))] for t in range(self.t_max)]) \
                .reshape(len(stored_funda) * self.t_max)
            second_index = np.array([stored_funda for t in range(self.t_max)]) \
                .reshape(len(stored_funda) * self.t_max)
            multi_index = [first_index, second_index]

            self.dftemp = pd.DataFrame(np.zeros((len(stored_funda) * self.t_max, self.n + 1)), index=multi_index)

    def clear_all(self):
        self.prices = np.zeros((self.t_max + 1, self.n))
        self.wages = np.zeros(self.t_max + 1)
        self.prices_net = np.zeros((self.t_max + 1, self.n))
        self.prods = np.zeros((self.t_max + 1, self.n))
        self.targets = np.zeros((self.t_max + 1, self.n))
        self.stocks = np.zeros((self.t_max + 1, self.n))
        self.profits = np.zeros((self.t_max + 1, self.n))
        self.balance = np.zeros((self.t_max + 1, self.n + 1))
        self.cashflow = np.zeros((self.t_max + 1, self.n))
        self.tradeflow = np.zeros((self.t_max + 1, self.n + 1))
        self.supply = np.zeros((self.t_max + 1, self.n + 1))
        self.demand = np.zeros((self.t_max + 1, self.n + 1))
        self.tradereal = np.zeros((self.t_max + 1, self.n + 1))
        self.s_vs_d = np.zeros((self.t_max + 1, self.n + 1))
        self.b_vs_c = np.zeros(self.t_max + 1)
        self.Q_real = np.zeros((self.t_max + 1, self.n + 1, self.n + 1))
        self.Q_demand = np.zeros((self.t_max + 1, self.n + 1, self.n + 1))
        self.mu = np.zeros(self.t_max + 1)
        self.budget = np.zeros(self.t_max + 1)
        self.utility = np.zeros(self.t_max + 1)
        self.labour = np.zeros(self.t_max + 1)

    def time_t_minus(self, t):
        """
        Performs all the actions of step t-. First the household is updated
        (budget, consumption and labour offer). Then, firms forecast profits and balance for the current
        time and compute a production target accordingly. They then compute and post their needs (both in
        goods and labor) as to reach this target.
        :param t: current time step
        :return: side-effect
        """

        # Firms
        self.supply[t] = np.concatenate(([self.labour[t]], self.eco.firms.z * self.prods[t] + self.stocks[t]))

        self.targets[t + 1] = self.eco.firms.compute_targets(self.prices[t],
                                                             self.Q_demand[t - 1],
                                                             self.supply[t],
                                                             self.prods[t]
                                                             )
        self.Q_demand[t, 1:] = self.eco.firms.compute_demands_firms(self.targets[t + 1],
                                                                    self.prices[t],
                                                                    self.prices_net[t],
                                                                    self.eco.q,
                                                                    self.eco.b,
                                                                    self.eco.lamb_a,
                                                                    )

    def time_t(self, t):
        """
        Performs all the actions of step t. First, an exchange period takes place where both firms and
        households may buy goods (including labor) in accordance to the supply constraint. Then, firms can
        compute their real profit and balance and update their prices (including wage) accordingly for the
        next time-step.
        :param t: current time-step
        :return: side-effect
        """

        self.demand[t] = np.sum(self.Q_demand[t], axis=0)
        # Supply constraint
        self.s_vs_d[t] = np.clip(self.supply[t] / self.demand[t], None, 1)  # =1 if supply >= constraint

        # Real work according to the labour supply constraint and associated budget
        self.Q_real[t, 1:, 0] = self.Q_demand[t, 1:, 0] * self.s_vs_d[t, 0]
        self.budget[t] = self.budget_res[t - 1] + np.sum(self.Q_real[t, 1:, 0])

        # Budget constraint
        offered_cons = self.Q_demand[t, 0, 1:] * self.s_vs_d[t, 1:]
        self.b_vs_c[t], self.Q_real[t, 0, 1:], self.budget_res[t] = self.eco.house.budget_constraint(self.budget[t],
                                                                                                     self.prices[t],
                                                                                                     offered_cons
                                                                                                     )

        # Real trades according to the supply constraint
        diag = np.diag(
            self.s_vs_d[t, 1:] + offered_cons * (1 - self.b_vs_c[t]) / np.sum(self.Q_demand[t, 1:, 1:], axis=0))

        self.Q_real[t, 1:, 1:] = np.clip(np.matmul(self.Q_demand[t, 1:, 1:],
                                                   diag),
                                         None,
                                         self.Q_demand[t, 1:, 1:])
        # print(self.Q_real[t])
        self.tradereal[t] = np.sum(self.Q_real[t], axis=0)

        # Prices and wage update
        self.profits[t], self.balance[t], self.cashflow[t], self.tradeflow[t] = \
            self.eco.firms.compute_profits_balance(self.prices[t],
                                                   self.Q_real[t],
                                                   self.supply[t],
                                                   self.demand[t]
                                                   )

        self.wages[t] = self.eco.firms.update_wages(self.balance[t, 0], self.tradeflow[t, 0])
        self.utility[t] = self.eco.house.utility(self.Q_real[t, 0, 1:], self.Q_real[t, 1:, 0])

    def time_t_plus(self, t):
        """
        Performs all the actions of step t+. Production for the next time-step starts and inventories
        are compiled.
        :param t: current time-step
        :return: side-effect
        """

        self.prices[t + 1] = self.eco.firms.update_prices(self.prices[t],
                                                          self.profits[t],
                                                          self.balance[t],
                                                          self.cashflow[t],
                                                          self.tradeflow[t]
                                                          ) / self.wages[t]

        self.budget[t] = self.budget[t] / self.wages[t]
        self.budget_res[t] = np.clip(self.budget_res[t], 0, None) / self.wages[
            t]  # Clipping to avoid negative almost zero values
        self.prices_net[t + 1] = self.eco.compute_p_net(self.prices[t + 1])

        self.prods[t + 1] = self.eco.production_function(self.Q_real[t, 1:, :])
        self.stocks[t + 1] = self.eco.firms.update_stocks(self.supply[t, 1:],
                                                          self.tradereal[t, 1:]
                                                          )

        self.mu[t], self.Q_demand[t + 1, 0, 1:], self.labour[t + 1] = \
            self.eco.house.compute_demand_cons_labour_supply(self.budget_res[t],
                                                             self.prices[t + 1],
                                                             )

    def discrete_dynamics(self, p0, w0, g0, t1, s0, B0):
        self.clear_all()
        # Initial conditions at t=0
        # Household
        self.wages[0] = w0
        self.budget_res[0] = B0 / w0

        self.prods[1] = g0
        self.stocks[1] = s0
        self.prices[1] = p0 / w0
        self.prices_net[1] = self.eco.compute_p_net(self.prices[1])
        self.mu[0], self.Q_demand[1, 0, 1:], self.labour[1] = \
            self.eco.house.compute_demand_cons_labour_supply(self.budget_res[0],
                                                             self.prices[1]
                                                             )

        self.supply[1] = np.concatenate([[self.labour[1]], self.eco.firms.z * g0 + s0])

        # Firms

        self.targets[2] = t1
        self.Q_demand[1, 1:] = self.eco.firms.compute_demands_firms(self.targets[2],
                                                                    self.prices[1],
                                                                    self.prices_net[1],
                                                                    self.eco.q,
                                                                    self.eco.b,
                                                                    self.eco.lamb_a,
                                                                    )

        self.time_t(1)
        self.time_t_plus(1)
        if self.store:
            self.dftemp.loc[str(1), 'Wage/Prices'] = np.concatenate(([self.wages[1]], self.prices[1]))
            self.dftemp.loc[str(1), 'Labour/Productions'] = np.concatenate(([self.labour[1]], self.prods[1]))
            self.dftemp.loc[str(1), 'Budget/Stocks'] = np.concatenate(([self.budget_res[1]], self.stocks[1]))
            self.dftemp.loc[str(1), 'Mu/Targets'] = np.concatenate(([self.mu[1]], self.targets[1]))
            self.dftemp.loc[str(1), 'Cons'] = np.concatenate(([np.nan], self.Q_real[1, 0, 1:]))
            self.dftemp.loc[str(1), 'Demand'] = self.demand[1]
        t = 2
        while t < self.t_max:
            # print(t)
            self.time_t_minus(t)
            self.time_t(t)
            self.time_t_plus(t)
            if self.store:
                self.dftemp.loc[str(t), 'Wage/Prices'] = np.concatenate(([self.wages[t]], self.prices[t]))
                self.dftemp.loc[str(t), 'Labour/Productions'] = np.concatenate(([self.labour[t]], self.prods[t]))
                self.dftemp.loc[str(t), 'Budget/Stocks'] = np.concatenate(([self.budget_res[t]], self.stocks[t]))
                self.dftemp.loc[str(t), 'Mu/Targets'] = np.concatenate(([self.mu[t]], self.targets[t]))
                self.dftemp.loc[str(t), 'Cons'] = np.concatenate(([np.nan], self.Q_real[t, 0, 1:]))
                self.dftemp.loc[str(t), 'Demand'] = self.demand[t]
            t += 1
        if self.store:
            self.dftemp.drop('0')
            self.dftemp.to_hdf(str(self.store) + '_dyn.h5', key='df', mode='w')

    def save_plot(self, dyn_name, k=None, from_eq=False):
        with mpl.rc_context(rc={'interactive': False}):
            plotter = gc(self)
            plotter.plotFirms(from_eq=from_eq, k=k)
            plotter.plotHousehold()

    def find_crises(self):
        return len(find_peaks(np.abs(self.prices[1:, 0]), distance=self.t_max//10)[0]) > 1
