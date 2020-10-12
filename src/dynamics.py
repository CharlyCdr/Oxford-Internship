import warnings

from numba import jit

warnings.simplefilter("ignore")

import numpy as np
from exception import InputError


class Dynamics(object):

    def __init__(self, e, t_max, store=None):
        self.eco = e
        self.t_max = t_max
        self.n = self.eco.n
        self.prices = np.zeros((t_max + 1, self.n))
        self.wages = np.zeros(t_max + 1)
        self.prices_net = np.zeros((t_max + 1, self.n))
        self.prods = np.zeros(self.n)
        self.targets = np.zeros(self.n)
        self.stocks = np.zeros((t_max + 1, self.n))
        self.profits = np.zeros(self.n)
        self.balance = np.zeros(self.n + 1)
        self.cashflow = np.zeros(self.n)
        self.tradeflow = np.zeros(self.n + 1)
        self.supply = np.zeros(self.n + 1)
        self.demand = np.zeros(self.n + 1)
        self.tradereal = np.zeros(self.n + 1)
        self.s_vs_d = np.zeros(self.n + 1)
        self.b_vs_c = 0
        self.Q_real = np.zeros((t_max + 1, self.n + 1, self.n + 1))
        self.Q_demand = np.zeros((t_max + 1, self.n + 1, self.n + 1))
        self.mu = np.zeros(t_max + 1)
        self.budget = np.zeros(t_max + 1)
        self.budget_res = 0
        self.labour = np.zeros(t_max + 1)

        self.store = store

        self.p0 = None
        self.w0 = None
        self.g0 = None
        self.t1 = None
        self.s0 = None
        self.B0 = None

        self.run_with_current_ic = False
        self.rho = 1

    def clear_all(self, t_max=None):
        self.prices = np.zeros((self.t_max + 1, self.n))
        self.wages = np.zeros(self.t_max + 1)
        self.prices_net = np.zeros(self.n)
        self.prods = np.zeros(self.n)
        self.targets = np.zeros(self.n)
        self.stocks = np.zeros((self.t_max + 1, self.n))
        self.profits = np.zeros(self.n)
        self.balance = np.zeros(self.n + 1)
        self.cashflow = np.zeros(self.n)
        self.tradeflow = np.zeros(self.n + 1)
        self.supply = np.zeros(self.n + 1)
        self.demand = np.zeros(self.n + 1)
        self.tradereal = np.zeros(self.n + 1)
        self.s_vs_d = np.zeros(self.n + 1)
        self.b_vs_c = 0
        self.Q_real = np.zeros((self.t_max + 1, self.n + 1, self.n + 1))
        self.Q_demand = np.zeros((self.t_max + 1, self.n + 1, self.n + 1))
        self.mu = np.zeros(self.t_max + 1)
        self.budget = np.zeros(self.t_max + 1)
        self.budget_res = 0
        self.labour = np.zeros(self.t_max + 1)

    def update_tmax(self, t_max):
        self.clear_all(t_max)
        self.store = self.store

        self.p0 = None
        self.w0 = None
        self.g0 = None
        self.t1 = None
        self.s0 = None
        self.B0 = None

        self.run_with_current_ic = False

    def update_eco(self, e):
        self.eco = e

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
        self.supply = np.concatenate(([self.labour[t]], self.eco.firms.z * self.prods + self.stocks[t]))

        self.targets = self.eco.firms.compute_targets(self.prices[t],
                                                      self.Q_demand[t - 1],
                                                      self.supply,
                                                      self.prods
                                                      )
        self.Q_demand[t, 1:] = self.eco.firms.compute_demands_firms(self.targets,
                                                                    self.prices[t],
                                                                    self.prices_net,
                                                                    self.eco.q,
                                                                    self.eco.b,
                                                                    self.eco.lamb_a,
                                                                    self.eco.j_a,
                                                                    self.eco.a_a,
                                                                    self.eco.zeros_j_a,
                                                                    self.n
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

        self.demand = np.sum(self.Q_demand[t], axis=0)
        # Supply constraint
        self.s_vs_d = np.clip(self.supply / self.demand, None, 1)  # =1 if supply >= constraint

        # Real work according to the labour supply constraint and associated budget
        self.Q_real[t, 1:, 0] = self.Q_demand[t, 1:, 0] * self.s_vs_d[0]
        self.budget[t] = self.budget_res + np.sum(self.Q_real[t, 1:, 0])

        # Budget constraint
        offered_cons = self.Q_demand[t, 0, 1:] * self.s_vs_d[1:]
        self.b_vs_c, self.Q_real[t, 0, 1:], self.budget_res = self.eco.house.budget_constraint(self.budget[t],
                                                                                               self.prices[t],
                                                                                               offered_cons
                                                                                               )

        # Real trades according to the supply constraint
        diag = np.diag(np.clip((self.supply[1:] - self.rho * self.Q_real[t, 0, 1:]) / (self.demand[1:] - self.rho * self.Q_demand[t, 0, 1:]),
                               None, 1))

        self.Q_real[t, 1:, 1:] = np.matmul(self.Q_demand[t, 1:, 1:],
                                                   diag)
        # print(self.Q_real[t])
        self.tradereal = np.sum(self.Q_real[t], axis=0)

        # Prices and wage update
        self.profits, self.balance, self.cashflow, self.tradeflow = \
            self.eco.firms.compute_profits_balance(self.prices[t],
                                                   self.Q_real[t],
                                                   self.supply,
                                                   self.demand
                                                   )

        self.wages[t] = self.eco.firms.update_wages(self.balance[0], self.tradeflow[0])
        # self.utility[t] = self.eco.house.utility(self.Q_real[t, 0, 1:], self.Q_real[t, 1:, 0])

    def time_t_plus(self, t):
        """
        Performs all the actions of step t+. Production for the next time-step starts and inventories
        are compiled.
        :param t: current time-step
        :return: side-effect
        """

        self.prices[t + 1] = self.eco.firms.update_prices(self.prices[t],
                                                          self.profits,
                                                          self.balance,
                                                          self.cashflow,
                                                          self.tradeflow
                                                          ) / self.wages[t]

        self.budget[t] = self.budget[t] / self.wages[t]
        self.budget_res = np.clip(self.budget_res, 0, None) / self.wages[t]
        # Clipping to avoid negative almost zero values
        self.prices_net = self.eco.compute_p_net(self.prices[t + 1])

        self.prods = self.eco.production_function(self.Q_real[t, 1:, :])
        self.stocks[t + 1] = self.eco.firms.update_stocks(self.supply[1:],
                                                          self.tradereal[1:]
                                                          )

        self.mu[t], self.Q_demand[t + 1, 0, 1:], self.labour[t + 1] = \
            self.eco.house.compute_demand_cons_labour_supply(self.budget_res,
                                                             self.prices[t + 1],
                                                             )

    def set_initial_conditions(self, p0, w0, g0, t1, s0, B0):
        self.p0 = p0
        self.w0 = w0
        self.g0 = g0
        self.t1 = t1
        self.s0 = s0
        self.B0 = B0
        self.run_with_current_ic = False

    def set_initial_price(self, p0):
        self.p0 = p0
        self.run_with_current_ic = False

    def set_initial_wage(self, w0):
        self.w0 = w0
        self.run_with_current_ic = False

    def set_initial_prod(self, g0):
        self.g0 = g0
        self.run_with_current_ic = False

    def set_initial_target(self, t1):
        self.t1 = t1
        self.run_with_current_ic = False

    def set_initial_stock(self, s0):
        self.s0 = s0
        self.run_with_current_ic = False

    def set_initial_budget(self, B0):
        self.B0 = B0
        self.run_with_current_ic = False

    def discrete_dynamics(self):
        # if not self.p0 or not self.w0 or not self.g0 or not self.s0 or not self.B0 or not self.t1:
        #     raise InputError("Inital conditions must be prescribed before running the simulation. Please use "
        #                      "the set_initial_conditions method.")

        self.clear_all()
        self.wages[0] = self.w0
        self.budget_res = self.B0 / self.w0

        self.prods = self.g0
        self.stocks[1] = self.s0
        self.prices[1] = self.p0 / self.w0
        self.prices_net = self.eco.compute_p_net(self.prices[1])
        self.mu[0], self.Q_demand[1, 0, 1:], self.labour[1] = \
            self.eco.house.compute_demand_cons_labour_supply(self.budget_res,
                                                             self.prices[1]
                                                             )

        self.supply = np.concatenate([[self.labour[1]], self.eco.firms.z * self.g0 + self.s0])

        # Firms

        self.targets = self.t1
        self.Q_demand[1, 1:] = self.eco.firms.compute_demands_firms(self.targets,
                                                                    self.prices[1],
                                                                    self.prices_net,
                                                                    self.eco.q,
                                                                    self.eco.b,
                                                                    self.eco.lamb_a,
                                                                    self.eco.j_a,
                                                                    self.eco.a_a,
                                                                    self.eco.zeros_j_a,
                                                                    self.n
                                                                    )

        self.time_t(1)
        self.time_t_plus(1)
        t = 2
        while t < self.t_max:
            # print(t)
            self.time_t_minus(t)
            self.time_t(t)
            self.time_t_plus(t)
            t += 1

        self.run_with_current_ic = True
        self.prods = self.g0
        self.targets = self.t1
        self.budget_res = self.B0 / self.w0

    @staticmethod
    @jit
    def compute_prods(e, Q_real, tmax, n, g0):
        prods = np.zeros((tmax + 1, n))
        prods[1] = g0
        for t in range(1, tmax - 1):
            prods[t + 1, :] = e.production_function(Q_real[t, 1:, :])
        return prods

    @staticmethod
    @jit
    def compute_profits_balance_cashflow_tradeflow(e, Q_real, Q_demand, prices, prods, stocks, labour, tmax, n):
        supply_goods = e.firms.z * prods + stocks
        demand = np.sum(Q_demand, axis=1)
        profits, balance, cashflow, tradeflow = np.zeros((tmax + 1, n)), np.zeros((tmax + 1, n + 1)), \
                                                np.zeros((tmax + 1, n)), np.zeros((tmax + 1, n + 1))
        for t in range(1, tmax):
            supply_t = np.concatenate(([labour[t]], supply_goods[t]))
            profits[t], balance[t], cashflow[t], tradeflow[t] = e.firms.compute_profits_balance(prices[t],
                                                                                                Q_real[t],
                                                                                                supply_t,
                                                                                                demand[t]
                                                                                                )
        return profits, balance, cashflow, tradeflow

    @staticmethod
    @jit
    def compute_utility(e, Q_real, tmax):
        utility = np.zeros(tmax + 1)
        for t in range(1, tmax):
            utility[t] = e.house.utility(Q_real[t, 0, 1:], Q_real[t, 1:, 0])
        return utility

    @staticmethod
    @jit
    def compute_targets(e, Q_demand, prices, tmax, n):
        targets = np.zeros((tmax + 1, n))
