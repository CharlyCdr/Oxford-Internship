import warnings

from numba import jit
from scipy.signal import periodogram, find_peaks
from scipy.special.cython_special import binom

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
from exception import InputError
from tqdm import tqdm_notebook as tqdm


class Dynamics(object):

    def __init__(self, e, t_max, step_size=None, boost=None, nu=None, store=None):
        self.eco = e
        self.t_max = t_max
        self.n = self.eco.n
        if step_size:
            self.step_s = step_size
        else:
            eps = self.eco.get_eps_cal()
            if eps >= 5:
                self.step_s = 1
            elif 5 > eps >= 0.1:
                self.step_s = 0.1
            elif eps < 0:
                self.step_s = 1
            else:
                self.step_s = np.power(10, np.floor(np.log10(np.abs(eps))))
        self.prices = np.zeros((int((t_max + 1) / self.step_s), self.n))
        self.wages = np.zeros(int((t_max + 1) / self.step_s))
        self.prices_net = np.zeros((int((t_max + 1) / self.step_s), self.n))
        self.prods = np.zeros((int((t_max + 1) / self.step_s), self.n))
        self.targets = np.zeros(self.n)
        self.stocks = np.zeros((int((t_max + 1) / self.step_s), self.n, self.n))
        self.gains = np.zeros((int((t_max + 1) / self.step_s), self.n))
        self.losses = np.zeros((int((t_max + 1) / self.step_s), self.n))
        self.supply = np.zeros((int((t_max + 1) / self.step_s), self.n + 1))
        self.demand = np.zeros((int((t_max + 1) / self.step_s), self.n + 1))
        self.tradereal = np.zeros(self.n + 1)
        self.q_exchange = np.zeros((self.n + 1, self.n + 1))
        self.q_demand = np.zeros((int((t_max + 1) / self.step_s), self.n + 1, self.n + 1))
        self.q_opt = np.zeros((self.n, self.n + 1))
        self.q_prod = np.zeros((self.n, self.n + 1))
        self.q_used = np.zeros((self.n, self.n + 1))
        self.budget = np.zeros(int((t_max + 1) / self.step_s))
        self.budget_res = 0
        self.labour = np.zeros(int((t_max + 1) / self.step_s))

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
        self.prices = np.zeros((int((self.t_max + 1) / self.step_s), self.n))
        self.wages = np.zeros(int((self.t_max + 1) / self.step_s))
        self.prices_net = np.zeros(self.n)
        self.prods = np.zeros((int((self.t_max + 1) / self.step_s), self.n))
        self.targets = np.zeros(self.n)
        self.stocks = np.zeros((int((self.t_max + 1) / self.step_s), self.n, self.n))
        self.gains = np.zeros((int((self.t_max + 1) / self.step_s), self.n))
        self.losses = np.zeros((int((self.t_max + 1) / self.step_s), self.n))
        self.supply = np.zeros((int((self.t_max + 1) / self.step_s), self.n + 1))
        self.demand = np.zeros((int((self.t_max + 1) / self.step_s), self.n + 1))
        self.tradereal = np.zeros(self.n + 1)
        self.q_exchange = np.zeros((self.n + 1, self.n + 1))
        self.q_demand = np.zeros((int((self.t_max + 1) / self.step_s), self.n + 1, self.n + 1))
        self.q_opt = np.zeros((self.n, self.n + 1))
        self.q_prod = np.zeros((self.n, self.n + 1))
        self.q_used = np.zeros((self.n, self.n + 1))
        self.budget = np.zeros(int((self.t_max + 1) / self.step_s))
        self.budget_res = 0
        self.labour = np.zeros(int((self.t_max + 1) / self.step_s))

    def update_tmax(self, t_max):
        self.clear_all(t_max)
        self.store = self.store
        self.run_with_current_ic = False

    def update_step_size(self, step_size):
        self.step_s = step_size
        self.clear_all(self.t_max)
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
                                                      self.prods[t] + self.boost,
                                                      self.step_s
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
        self.q_demand[t, 1:, 1:] = np.maximum(
            self.q_opt[:, 1:] - (self.stocks[t] - np.diagonal(self.stocks[t]) * np.eye(self.n)),
            0)

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

        s_vs_d = np.minimum(self.supply[t] / self.demand[t], 1)  # =1 if supply >= constraint

        self.q_exchange[:, 1:] = np.matmul(self.q_demand[t, :, 1:], np.diag(s_vs_d[1:]))

        self.q_exchange[0, 1:] = self.q_exchange[0, 1:] * np.minimum(1, self.budget[t] / (
            np.dot(self.q_exchange[0, 1:], self.prices[t])))

        self.budget_res = self.budget[t] - np.dot(self.prices[t], self.q_exchange[0, 1:])

        self.q_prod[:, 0] = self.q_exchange[1:, 0]
        self.q_prod[:, 1:] = self.q_exchange[1:, 1:] + np.minimum(
            self.stocks[t] - np.diag(self.stocks[t]) * np.eye(self.n), self.q_opt[:, 1:])

        self.tradereal = np.sum(self.q_exchange, axis=0)

        self.gains[t], self.losses[t] = self.prices[t] * self.tradereal[1:], \
                                        np.matmul(self.q_exchange[1:, :],
                                                  np.concatenate(
                                                      ([1], self.prices[t]))
                                                  )

        self.wages[t + 1] = self.eco.firms.update_wages(self.supply[t, 0] - self.demand[t, 0],
                                                        self.supply[t, 0] + self.demand[t, 0],
                                                        self.step_s)

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
                                                          self.supply[t] + self.demand[t],
                                                          self.step_s
                                                          ) / self.wages[t + 1]

        self.budget[t] = self.budget[t] / self.wages[t + 1]
        self.budget_res = np.maximum(self.budget_res, 0) / self.wages[t + 1]
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

        self.stocks[t + 1] = np.matmul(self.stocks[t + 1], np.diag(np.exp(- self.eco.firms.sigma * self.step_s)))

        self.q_demand[t + 1, 0, 1:], self.labour[t + 1] = \
            self.eco.house.compute_demand_cons_labour_supply(self.budget_res,
                                                             self.prices[t + 1],
                                                             self.supply[t, 0],
                                                             self.demand[t, 0],
                                                             self.nu,
                                                             self.step_s
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
                                                             self.nu,
                                                             self.step_s
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
        self.q_demand[1, 1:, 1:] = np.maximum(
            self.q_opt[:, 1:] - (self.stocks[1] - np.diagonal(self.stocks[1]) * np.eye(self.n)),
            0)

        self.time_t(1)
        self.time_t_plus(1)
        t = 2
        pbar = tqdm(total=self.t_max + 1)
        pbar.update(2 * self.step_s)
        while t < int((self.t_max + 1) / self.step_s - 1):
            # print(t)
            self.time_t_minus(t)
            self.time_t(t)
            self.time_t_plus(t)
            t += 1
            pbar.update(self.step_s)

        self.run_with_current_ic = True

    def norm_prices_prods_stocks(self):
        if not self.eco.p_eq or not self.eco.p_eq:
            self.eco.compute_eq()

        dfp = pd.DataFrame(self.prices[1:-1] - self.eco.p_eq, columns=['p' + str(i) for i in range(self.n)])
        dfg = pd.DataFrame(self.prices[1:-1] - self.eco.g_eq, columns=['g' + str(i) for i in range(self.n)])
        dfs = pd.DataFrame(self.prices[1:-1], columns=['s' + str(i) for i in range(self.n)])
        df = pd.concat([dfp, dfg, dfs], axis=1)
        df = np.linalg.norm(df, axis=1)
        return df

    @staticmethod
    def rolling_diff(data):
        """
        Function used for classifying the long-term behaviour of the prices in  a
        single simulation: rolling diff version.
        :param sim: Dynamics object
        :param threshold: float, threshold for the function's precision
        :return: Bool, for "prices converge" statement
        """
        t_diff = []
        for t in range(1, len(data) - 10 + 1):
            t_diff.append(np.amax(data[- t - 10:- t]) - np.amin(data[- t - 10:- t]))
        df_t_diff = pd.DataFrame(t_diff[::-1])
        return df_t_diff.apply(lambda x: x.is_monotonic_decreasing)[0], df_t_diff.iloc[-1] < 10e-8

    @staticmethod
    def fisher_test(data):
        freq, dft = periodogram(data, fs=1)
        q = int((len(data) - 1) / 2)
        stat = max(dft) / np.sum(dft)
        b = int(1 / stat)
        binom_vec = np.vectorize(lambda j: binom(q, j))
        j_vec = np.arange(b + 1)
        p_value = 1 - np.sum(np.power(-1, j_vec) * binom_vec(q, j_vec) * np.power(1 - j_vec * stat, q - 1))
        return p_value

    def detect_periodicity(self, data):
        """
        Function used to determine if prices are periodic using a Fisher test at level 0.001.
        :return: True if periodic, False otherwise
        """
        return self.fisher_test(data) < 0.001

    def detect_convergent(self, data):
        """
        :param data:
        :return:
        """
        return self.rolling_diff(data)

    def detect_divergent(self, data):
        """
        Function used to determine if prices diverge.
        :param sim: Dynamics object
        :return: True if prices diverge, False otherwise
        """
        if np.isnan(data).any():
            return True
        else:
            t_diff = []
            for t in range(1, len(data) - 10 + 1):
                t_diff.append(np.amax(data[- t - 10:- t]) - np.amin(data[- t - 10:- t]))
            df_t_diff = pd.DataFrame(t_diff[::-1])
            if df_t_diff.apply(lambda x: x.is_monotonic_increasing)[0] and \
                    df_t_diff.apply(lambda x: x.is_monotonic_decreasing)[0]:
                return False
            else:
                return df_t_diff.apply(lambda x: x.is_monotonic_increasing)[0]

    def detect_crises(sim):
        """
        Function used to determine if prices converge or diverge.
        :param sim: Dynamics object
        :return: True in converges, False otherwise
        """
        fact_norm = float(max(sim.prices[-1000:-1]))
        peaks_minus, peaks_properties_minus = find_peaks(-sim.prices.T[0][-1000:-1] / fact_norm)
        peaks_plus, peaks_properties_plus = find_peaks(sim.prices.T[0][-1000:-1] / fact_norm)
        return np.average(sim.prices.T[0][-1000:-1][peaks_plus]) / np.average(
            sim.prices.T[0][-1000:-1][peaks_minus]) > 10e2

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
