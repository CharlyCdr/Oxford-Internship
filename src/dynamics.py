import warnings

from numba import jit

warnings.simplefilter("ignore")

import numpy as np
from exception import InputError


class Dynamics(object):

    def __init__(self, e, t_max, boost=None, nu=None, store=None):
        self.eco = e
        self.t_max = t_max
        self.n = self.eco.n
        self.prices = np.zeros((t_max + 1, self.n))
        self.wages = np.zeros(t_max + 1)
        self.prices_net = np.zeros((t_max + 1, self.n))
        self.prods = np.zeros((t_max + 1, self.n))
        self.targets = np.zeros(self.n)
        self.stocks = np.zeros((t_max + 1, self.n, self.n))
        self.gains = np.zeros((t_max + 1, self.n))
        self.losses = np.zeros((t_max + 1, self.n))
        self.supply = np.zeros((t_max + 1, self.n + 1))
        self.demand = np.zeros((t_max + 1, self.n + 1))
        self.tradereal = np.zeros(self.n + 1)
        self.q_exchange = np.zeros((self.n + 1, self.n + 1))
        self.q_demand = np.zeros((t_max + 1, self.n + 1, self.n + 1))
        self.q_opt = np.zeros((self.n, self.n + 1))
        self.q_prod = np.zeros((self.n, self.n + 1))
        self.q_used = np.zeros((self.n, self.n + 1))
        self.budget = np.zeros(t_max + 1)
        self.budget_res = 0
        self.labour = np.zeros(t_max + 1)
        self.boost = boost if boost else 0

        self.store = store

        self.p0 = None
        self.w0 = None
        self.g0 = None
        self.t1 = None
        self.s0 = None
        self.B0 = None

        self.run_with_current_ic = False
        self.nu = nu if nu else 1

    def clear_all(self, t_max=None):
        if t_max:
            self.t_max = t_max
        self.prices = np.zeros((self.t_max + 1, self.n))
        self.wages = np.zeros(self.t_max + 1)
        self.prices_net = np.zeros(self.n)
        self.prods = np.zeros((self.t_max + 1, self.n))
        self.targets = np.zeros(self.n)
        self.stocks = np.zeros((self.t_max + 1, self.n, self.n))
        self.gains = np.zeros((self.t_max + 1, self.n))
        self.losses = np.zeros((self.t_max + 1, self.n))
        self.supply = np.zeros((self.t_max + 1, self.n + 1))
        self.demand = np.zeros((self.t_max + 1, self.n + 1))
        self.tradereal = np.zeros(self.n + 1)
        self.q_exchange = np.zeros((self.n + 1, self.n + 1))
        self.q_demand = np.zeros((self.t_max + 1, self.n + 1, self.n + 1))
        self.q_opt = np.zeros((self.n, self.n + 1))
        self.q_prod = np.zeros((self.n, self.n + 1))
        self.q_used = np.zeros((self.n, self.n + 1))
        self.budget = np.zeros(self.t_max + 1)
        self.budget_res = 0
        self.labour = np.zeros(self.t_max + 1)

    def update_tmax(self, t_max):
        self.clear_all(t_max)
        self.store = self.store
        self.run_with_current_ic = False

    def update_eco(self, e):
        self.eco = e

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

    def time_t_minus(self, t):
        """
        Performs all the actions of step t-. First the household is updated
        (budget, consumption and labour offer). Then, firms forecast profits and balance for the current
        time and compute a production target accordingly. They then compute and post their needs (both in
        goods and labor) as to reach this target.
        :param t: current time step
        :return: side-effect
        """

        self.supply[t] = np.concatenate(
            ([self.labour[t]], self.eco.firms.z * self.prods[t] + np.diagonal(self.stocks[t])))

        self.targets = self.eco.firms.compute_targets(self.prices[t],
                                                      self.q_demand[t - 1],
                                                      self.supply[t],
                                                      self.prods[t] + self.boost
                                                      )
        self.q_opt = self.eco.firms.compute_optimal_quantities_firms(self.targets,
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

        self.q_demand[t, 1:, 0] = self.q_opt[:, 0]
        self.q_demand[t, 1:, 1:] = np.clip(
            self.q_opt[:, 1:] - (self.stocks[t] - np.diagonal(self.stocks[t]) * np.eye(self.n)),
            0,
            None)

    def time_t(self, t):
        """
        Performs all the actions of step t. First, an exchange period takes place where both firms and
        households may buy goods (including labor) in accordance to the supply constraint. Then, firms can
        compute their real profit and balance and update their prices (including wage) accordingly for the
        next time-step.
        :param t: current time-step
        :return: side-effect
        """

        self.q_exchange[1:, 0] = self.q_demand[t, 1:, 0] * np.minimum(1, self.labour[t] / np.sum(
            self.q_demand[t, 1:, 0]))

        self.budget[t] = self.budget_res + np.sum(self.q_exchange[1:, 0])

        self.demand[t] = np.sum(self.q_demand[t], axis=0)

        s_vs_d = np.clip(self.supply[t] / self.demand[t], None, 1)  # =1 if supply >= constraint

        self.q_exchange[:, 1:] = np.matmul(self.q_demand[t, :, 1:], np.diag(s_vs_d[1:]))

        self.q_exchange[0, 1:] = self.q_exchange[0, 1:] * np.minimum(1, self.budget[t] / (
                np.dot(self.prices[t], self.q_exchange[0, 1:])))

        self.budget_res = self.budget[t] - np.dot(self.prices[t], self.q_exchange[0, 1:])

        self.q_prod[:, 0] = self.q_exchange[1:, 0]
        self.q_prod[:, 1:] = self.q_exchange[1:, 1:] + np.minimum(
            self.stocks[t] - np.diag(self.stocks[t]) * np.eye(self.n), self.q_opt[:, 1:])

        self.tradereal = np.sum(self.q_exchange, axis=0)

        self.gains[t], self.losses[t] = self.prices[t] * self.tradereal[1:], np.matmul(self.q_exchange[1:, :],
                                                                                       np.concatenate(
                                                                                           ([1], self.prices[t]))
                                                                                       )

        self.wages[t + 1] = self.eco.firms.update_wages(self.supply[t, 0] - self.demand[t, 0],
                                                        self.supply[t, 0] + self.demand[t, 0])

    def time_t_plus(self, t):
        """
        Performs all the actions of step t+. Production for the next time-step starts and inventories
        are compiled.
        :param t: current time-step
        :return: side-effect
        """

        self.prices[t + 1] = self.eco.firms.update_prices(self.prices[t],
                                                          self.gains[t] - self.losses[t],
                                                          self.supply[t] - self.demand[t],
                                                          self.gains[t] + self.losses[t],
                                                          self.supply[t] + self.demand[t]
                                                          ) / self.wages[t + 1]

        self.budget[t] = self.budget[t] / self.wages[t + 1]
        self.budget_res = np.clip(self.budget_res, 0, None) / self.wages[t + 1]
        # Clipping to avoid negative almost zero values
        self.prices_net = self.eco.compute_p_net(self.prices[t + 1])

        self.prods[t + 1] = self.eco.production_function(self.q_prod)

        self.q_used = (self.eco.q == 0) * np.matmul(np.diag(np.nanmin(np.divide(self.q_prod, self.eco.j_a), axis=1)),
                                                    self.eco.j_a) + (self.eco.q != 0) * self.q_prod

        stocks_no_diagonal = self.stocks[t] - np.diag(self.stocks[t]) * np.eye(self.n)

        self.stocks[t + 1] = (self.eco.q == 0) * ((np.heaviside(stocks_no_diagonal - self.q_opt[:, 1:], 1)) * (
                stocks_no_diagonal - self.q_used[:, 1:])
                                                  + (1 - np.heaviside(stocks_no_diagonal - self.q_opt[:, 1:], 1)) * (
                                                          self.q_prod[:, 1:] - self.q_used[:, 1:]))

        np.fill_diagonal(self.stocks[t + 1], self.supply[t, 1:] - self.tradereal[1:])

        self.stocks[t + 1] = np.matmul(self.stocks[t + 1], np.diag(1 - self.eco.firms.sigma))

        self.q_demand[t + 1, 0, 1:], self.labour[t + 1] = \
            self.eco.house.compute_demand_cons_labour_supply(self.budget_res,
                                                             self.prices[t + 1],
                                                             self.supply[t, 0],
                                                             self.demand[t, 0],
                                                             self.nu
                                                             )

    def discrete_dynamics(self):
        # if not self.p0 or not self.w0 or not self.g0 or not self.s0 or not self.B0 or not self.t1:
        #     raise InputError("Inital conditions must be prescribed before running the simulation. Please use "
        #                      "the set_initial_conditions method.")

        self.clear_all()
        self.wages[1] = self.w0
        self.budget_res = self.B0 / self.w0

        self.prods[1] = self.g0
        self.stocks[1] = self.s0
        self.prices[1] = self.p0 / self.w0
        self.prices_net = self.eco.compute_p_net(self.prices[1])
        self.q_demand[1, 0, 1:], self.labour[1] = \
            self.eco.house.compute_demand_cons_labour_supply(self.budget_res,
                                                             self.prices[1],
                                                             1,
                                                             1,
                                                             self.nu
                                                             )

        self.supply[1] = np.concatenate([[self.labour[1]], self.eco.firms.z * self.g0 + np.diagonal(self.s0)])

        # Firms

        self.targets = self.t1
        self.q_opt = self.eco.firms.compute_optimal_quantities_firms(self.targets,
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

        self.q_demand[1, 1:, 0] = self.q_opt[:, 0]
        self.q_demand[1, 1:, 1:] = np.clip(
            self.q_opt[:, 1:] - (self.stocks[1] - np.diagonal(self.stocks[1]) * np.eye(self.n)),
            0,
            None)

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
        # self.prods = self.g0
        # self.targets = self.t1
        # self.budget_res = self.B0 / self.w0

    @staticmethod
    @jit
    def reconstruct_prods(e, Q_real, tmax, n, g0):
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
