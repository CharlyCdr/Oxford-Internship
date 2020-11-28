import warnings
import numpy as np
import pandas as pd

from numba import jit
from scipy.signal import periodogram
from scipy.special.cython_special import binom

warnings.simplefilter("ignore")


class Dynamics(object):
    def __init__(self, e, t_max, step_size=None, lda=None, nu=None, store=None):
        self.eco = e  # Economy for which to run the simulations
        self.t_max = t_max  # End time of the simulation
        self.n = self.eco.n  # Number of firms
        self.step_s = step_size if step_size else 1  # Size of on time step

        # Initialization of time-series
        self.prices = np.zeros((int((t_max + 1) / self.step_s), self.n))
        self.prices_non_res = np.zeros((int((t_max + 1) / self.step_s), self.n))
        self.wages = np.zeros(int((t_max + 1) / self.step_s))
        self.prods = np.zeros((int((t_max + 1) / self.step_s), self.n))
        self.targets = np.zeros((int((t_max + 1) / self.step_s), self.n))
        self.stocks = np.zeros((int((t_max + 1) / self.step_s), self.n, self.n))
        self.gains = np.zeros((int((t_max + 1) / self.step_s), self.n))
        self.losses = np.zeros((int((t_max + 1) / self.step_s), self.n))
        self.supply = np.zeros((int((t_max + 1) / self.step_s), self.n + 1))
        self.demand = np.zeros((int((t_max + 1) / self.step_s), self.n + 1))
        self.tradereal = np.zeros(self.n + 1)
        self.q_exchange = np.zeros((int((t_max + 1) / self.step_s), self.n + 1, self.n + 1))
        self.q_demand = np.zeros((int((t_max + 1) / self.step_s), self.n + 1, self.n + 1))
        self.q_opt = np.zeros((self.n, self.n + 1))
        self.q_prod = np.zeros((self.n, self.n + 1))
        self.q_used = np.zeros((self.n, self.n + 1))
        self.budget = 0
        self.savings = 0
        self.labour = np.zeros(int((t_max + 1) / self.step_s))

        # Whether to store the dynamics in a h5 format
        self.store = store

        # Declare initial conditions instances
        self.p0 = None
        self.w0 = None
        self.g0 = None
        self.t1 = None
        self.s0 = None
        self.B0 = None

        self.run_with_current_ic = False
        self.nu = nu if nu else 1
        self.lda = lda if lda else 1

    def clear_all(self, t_max=None):
        """
        Clear every time-series in memory.
        :param t_max:
        :return: Emptied time-series instances.
        """
        if t_max:
            self.t_max = t_max
        self.prices = np.zeros((int((self.t_max + 1) / self.step_s), self.n))
        self.prices_non_res = np.zeros((int((self.t_max + 1) / self.step_s), self.n))
        self.wages = np.zeros(int((self.t_max + 1) / self.step_s))
        self.prods = np.zeros((int((self.t_max + 1) / self.step_s), self.n))
        self.targets = np.zeros((int((self.t_max + 1) / self.step_s), self.n))
        self.stocks = np.zeros((int((self.t_max + 1) / self.step_s), self.n, self.n))
        self.gains = np.zeros(self.n)
        self.losses = np.zeros(self.n)
        self.supply = np.zeros(self.n + 1)
        self.demand = np.zeros(self.n + 1)
        self.tradereal = np.zeros(self.n + 1)
        self.q_exchange = np.zeros((int((self.t_max + 1) / self.step_s), self.n + 1, self.n + 1))
        self.q_demand = np.zeros((int((self.t_max + 1) / self.step_s), self.n + 1, self.n + 1))
        self.q_opt = np.zeros((self.n, self.n + 1))
        self.q_prod = np.zeros((self.n, self.n + 1))
        self.q_used = np.zeros((self.n, self.n + 1))
        self.budget = 0
        self.savings = 0
        self.labour = np.zeros(int((self.t_max + 1) / self.step_s))

    # Setters for simulation parameters

    def update_tmax(self, t_max):
        self.clear_all(t_max)
        self.store = self.store
        self.run_with_current_ic = False

    def update_step_size(self, step_size):
        self.step_s = step_size
        self.clear_all(self.t_max)
        self.run_with_current_ic = False

    # Setters for the economy

    def update_eco(self, e):
        self.eco = e

    # Setters for initial conditions

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

    # Dynamical methods

    def planning(self, t):
        """
        Performs all the actions the Planning step. Firms forecast gains, losses, supplies and demands (in the
        compute_targets function) and compute the subsequent targets. Then, they post their demands (goods and labor)
        for the current period.
        :param t: current time step
        :return: side-effect
        """

        # (1) - (2) Forecasts and production targets
        self.supply = np.concatenate(
            ([self.labour[t]], self.eco.firms.z * self.prods[t] + np.diagonal(self.stocks[t])))

        self.targets[t + 1] = self.eco.firms.compute_targets(self.prices[t],
                                                             self.lda * self.q_demand[t - 1] +
                                                             (1 - self.lda) * self.q_exchange[t - 1],
                                                             self.supply,
                                                             self.prods[t],
                                                             self.step_s
                                                             )
        self.q_opt = self.eco.firms.compute_optimal_quantities(self.targets[t + 1],
                                                               self.prices[t],
                                                               self.eco
                                                               )

        # (3) Posting demands
        self.q_demand[t, 1:, 0] = self.q_opt[:, 0]
        self.q_demand[t, 1:, 1:] = np.maximum(
            self.q_opt[:, 1:] - (self.stocks[t] - np.diagonal(self.stocks[t]) * np.eye(self.n)),
            0)

    def exchanges_and_updates(self, t):
        """
        Performs all the actions the Exchanges & Trades step. Hiring occurs first, upon which the household gets paid.
        It allows it to compute its current budget and adapt its consumption target accordingly. Then, we move on to
        the trades where firms buy from one another and the household consumes (and computes its subsequent savings).
        Finally, everything is known for the firms to compute profits and balances and update prices and wage.
        :param t: current time-step
        :return: side-effect
        """

        # (1) Hiring and Wage payment
        self.q_exchange[t, 1:, 0] = self.q_demand[t, 1:, 0] * np.minimum(1, self.labour[t] / np.sum(
            self.q_demand[t, 1:, 0]))

        self.budget = self.savings + np.sum(self.q_exchange[t, 1:, 0])

        self.q_demand[t, 0, 1:] = self.q_demand[t, 0, 1:] * (self.nu + (1 - self.nu) *
                                                             np.minimum(1, self.budget /
                                                                        (self.savings + self.labour[t])))

        # (2) Trades
        self.demand = np.sum(self.q_demand[t], axis=0)

        self.q_exchange[t, :, 1:] = np.matmul(self.q_demand[t, :, 1:],
                                              np.diag(np.minimum(self.supply[1:] / self.demand[1:], 1))
                                              )

        self.q_exchange[t, 0, 1:] = self.q_exchange[t, 0, 1:] * np.minimum(1, self.budget / (
            np.dot(self.q_exchange[t, 0, 1:], self.prices[t])))

        self.savings = self.budget - np.dot(self.prices[t], self.q_exchange[t, 0, 1:])

        self.q_prod[:, 0] = self.q_exchange[t, 1:, 0]
        self.q_prod[:, 1:] = self.q_exchange[t, 1:, 1:] + np.minimum(
            self.stocks[t] - np.diag(self.stocks[t]) * np.eye(self.n), self.q_opt[:, 1:])

        self.tradereal = np.sum(self.q_exchange[t], axis=0)

        self.gains = self.prices[t] * self.tradereal[1:]
        self.losses = np.matmul(self.q_exchange[t, 1:, :], np.concatenate(([1], self.prices[t])))

        # (3) Prices and Wage updates
        self.wages[t + 1] = self.eco.firms.update_wages(self.supply[0] - self.demand[0],
                                                        self.supply[0] + self.demand[0],
                                                        self.step_s)

        self.prices[t + 1] = self.eco.firms.update_prices(self.prices[t],
                                                          self.gains - self.losses,
                                                          self.supply - self.demand,
                                                          self.gains + self.losses,
                                                          self.supply + self.demand,
                                                          self.step_s
                                                          )

    def production(self, t):
        """
        Performs all the actions of the Production Step. Production for the next time-step starts, inventories
        are compiled and monetary quantities are rescaled by the wage value. Then, the household then performs its
        optimization thus setting consumption targets and labour supply for the next period.
        :param t: current time-step
        :return: side-effect
        """

        # (1) Production starts
        self.prods[t + 1] = self.eco.production_function(self.q_prod)

        self.q_used = (self.eco.q == 0) * np.matmul(np.diag(np.nanmin(np.divide(self.q_prod, self.eco.j_a), axis=1)),
                                                    self.eco.j_a) + (self.eco.q != 0) * self.q_prod

        # (2) Inventory update
        self.stocks[t + 1] = (self.eco.q == 0) * (self.q_prod[:, 1:] - self.q_used[:, 1:])

        np.fill_diagonal(self.stocks[t + 1], self.supply[1:] - self.tradereal[1:])

        self.stocks[t + 1] = np.matmul(self.stocks[t + 1], np.diag(np.exp(- self.eco.firms.sigma * self.step_s)))

        # (3) Price rescaling
        self.prices[t + 1] = self.prices[t + 1] / self.wages[t + 1]
        self.budget = self.budget / self.wages[t + 1]
        self.savings = np.maximum(self.savings, 0) / self.wages[t + 1]
        # Clipping to avoid negative almost zero values

        # The household performs its optimization to set its consumption target and its labour supply for the next
        # period
        self.q_demand[t + 1, 0, 1:], self.labour[t + 1] = \
            self.eco.house.compute_demand_cons_labour_supply(self.savings,
                                                             self.prices[t + 1],
                                                             self.supply[0],
                                                             self.demand[0],
                                                             self.step_s
                                                             )

    def discrete_dynamics(self):
        """
        Main function to run the dynamics of the Network Economy ABM.
        :return: Side-effect
        """
        self.clear_all()

        # Setting initial conditions
        self.wages[1] = self.w0
        self.savings = self.B0 / self.w0

        self.prods[1] = self.g0
        self.stocks[1] = self.s0
        self.prices[1] = self.p0 / self.w0
        self.prices_non_res[1] = self.p0
        self.q_demand[1, 0, 1:], self.labour[1] = \
            self.eco.house.compute_demand_cons_labour_supply(self.savings,
                                                             self.prices[1],
                                                             1,
                                                             1,
                                                             self.step_s
                                                             )

        # Planning period with provided initial target t1.
        self.supply = np.concatenate([[self.labour[1]], self.eco.firms.z * self.g0 + np.diagonal(self.s0)])
        self.targets[2] = self.t1
        self.q_opt = self.eco.firms.compute_optimal_quantities(self.targets[2],
                                                               self.prices[1],
                                                               self.eco
                                                               )

        self.q_demand[1, 1:, 0] = self.q_opt[:, 0]
        self.q_demand[1, 1:, 1:] = np.maximum(
            self.q_opt[:, 1:] - (self.stocks[1] - np.diagonal(self.stocks[1]) * np.eye(self.n)),
            0)

        # Carrying on with Exchanges & Trades and Production with every needed quantities known.
        self.exchanges_and_updates(1)
        self.production(1)
        # End of first time-step
        t = 2
        while t < int((self.t_max + 1) / self.step_s - 1):
            self.planning(t)
            self.exchanges_and_updates(t)
            self.production(t)
            t += 1

        # The current information stocked in the dynamics class are in accordance with the provided initial conditions.
        self.run_with_current_ic = True

    # Classification methods

    def norm_prices_prods_stocks(self):
        """
        :return: A data-frame of prices, productions and diagonal stocks across time.
        """
        dfp = pd.DataFrame(self.prices[1:-1] - self.eco.p_eq, columns=['p' + str(i) for i in range(self.n)])
        dfg = pd.DataFrame(self.prods[1:-1] - self.eco.g_eq, columns=['g' + str(i) for i in range(self.n)])
        dfs = pd.DataFrame([np.diagonal(s) for s in self.stocks[1:-1]], columns=['s' + str(i) for i in range(self.n)])
        df = pd.concat([dfp, dfg, dfs], axis=1)
        df = df.apply(lambda x: np.linalg.norm(x), axis=1)
        return df

    @staticmethod
    def rolling_diff(data, step_back):
        """
        :param data: data-frame on which to perform the rolling diff.
        :param step_back: window on which to perform the rolling diff.
        :return: Rolling min-max diff of data on a given window.
        """
        t_diff = []
        for t in range(1, len(data) - step_back + 1):
            t_diff.append(np.amax(data[- t - step_back:- t]) - np.amin(data[- t - step_back:- t]))
        df_t_diff = pd.DataFrame(t_diff[::-1])
        return df_t_diff.apply(lambda x: x.is_monotonic_decreasing)[0], (df_t_diff.iloc[-1] < 10e-8)[0]

    @staticmethod
    def fisher_test(data):
        """
        Perform a Fisher test on data-frame to determine if iy oscillates.
        References: Ahdesmäki M, Lähdesmäki H, Pearson R, Huttunen H, Yli-Harja O.
                    Robust detection of periodic time series measured from biological systems.
                    BMC Bioinformatics. 2005;6:117. Published 2005 May 13. doi:10.1186/1471-2105-6-117
        :param data: data-frame on which to perform the Fisher test.
        :return: p-value computed for the Fisher test.
        """
        freq, dft = periodogram(data, fs=1)
        q = int((len(data) - 1) / 2)
        stat = max(dft) / np.sum(dft)
        b = int(1 / stat)
        binom_vec = np.vectorize(lambda j: binom(q, j))
        j_vec = np.arange(b + 1)
        p_value = 1 - np.sum(np.power(-1, j_vec) * binom_vec(j_vec) * np.power(1 - j_vec * stat, q - 1))
        return p_value

    def detect_periodicity(self, data):
        """
        Function used to determine whether data is periodic using a Fisher test at level 0.001. We exclude extreme
        oscillations along with constant functions.
        :param data: data for which to check periodicity.
        :return: True if periodic.
        """
        return self.fisher_test(data) < 0.001 and 10e8 > data.var() > 10e-8

    def detect_convergent(self, data):
        """
        Function to determine whether data is convergent using rolling min-max.
        :param data: data for which to check convergency.
        :return: 3-tuple: True if convergent, True is convergent and oscillating, True if convergent towards 0.
        """
        bools = self.rolling_diff(data, 10)
        return bools[0], bools[1], data.iloc[-1] < 10e-8

    @staticmethod
    def detect_divergent(data):
        """
        Function used to determine if data diverges using rolling min-max. We consider divergent data for which any of
        the rolling min-max value is nan.
        :param data: data for which to check divergency.
        :return: True if prices diverge, False otherwise
        """
        if np.isnan(data).any():
            return True
        else:
            t_diff = []
            for t in range(1, len(data) - 10 + 1):
                t_diff.append(np.amax(data[- t - 10:- t]) - np.amin(data[- t - 10:- t]))
            df_t_diff = pd.DataFrame(t_diff[::-1])
            return np.isnan(df_t_diff).any()[0] or (df_t_diff.iloc[-1] > 10e6)[0]

    def detect_crises(self, data):
        """
        ! Need to work on this function. !
        Function used to determine if data exhibits a crises-like patterns.
        :param data: data for which to check crises-like patterns.
        :return: True if data is both convergent and relaxes towards 0.
        """
        return self.fisher_test(data) < 0.001 and data.min() < 10e-4

    # Reconstruction methods

    @staticmethod
    @jit
    def compute_gains_losses_supplies_demand(e, q_demand, q_exchange, prices, prods, stocks, labour):
        """
        Reconstruction method to compute gains, losses, supplies and demands across time.
        :param e: economy class,
        :param q_demand: time-series of demand matrices,
        :param q_exchange: time-series of exchange matrices,
        :param prices: time-series of wage-rescaled prices,
        :param prods: time-series of production levels
        :param stocks: time-series of stocks,
        :param labour: time-series of labour supply.
        :return: Time-series of computed gains, losses, supplies and demands.
        """
        demands = np.sum(q_demand, axis=1)
        gains = np.zeros((len(prices), e.n))
        losses = np.zeros((len(prices), e.n))
        supplies = np.zeros((len(prices), e.n + 1))
        for i in range(len(prices)):
            gains[i] = prices[i] * np.sum(q_exchange[i, :, 1:], axis=0)
            losses[i] = np.matmul(q_exchange[i, 1:, :], np.concatenate(([1], prices[i])))
            supplies[i] = np.concatenate(([labour[i]], e.firms.z * prods[i] + np.diag(stocks[i])))
        return gains, losses, supplies, demands

    @staticmethod
    @jit
    def compute_utility_budget(e, q_exchange, prices, rescaling_factors, t_max, step_s, initial_savings):
        """
        Reconstruction method to compute utility and non-rescaled budget across time.
        :param e: economy class,
        :param q_exchange: time-series of exchange matrices,
        :param prices: time-series of prices,
        :param rescaling_factors: time-series of wages used to rescale prices,
        :param t_max: maximum number of time-steps,
        :param step_s: size of one time step,
        :param initial_savings: initial savings
        :return: Time-series for utility and non-rescaled budgets.
        """
        utility = np.zeros(int((t_max + 1) / step_s))
        budget = np.zeros(int((t_max + 1) / step_s))
        budget[0] = initial_savings
        for i in range(1, int((t_max + 1) / step_s) - 1):
            utility[i] = np.power(rescaling_factors[i + 1], e.house.omega_p / e.firms.omega) * \
                         np.dot(e.house.theta, q_exchange[i, 0, 1:]) - \
                         np.power(np.sum(q_exchange[i, 1:, 0]) / e.house.l_0, 1 + e.house.phi) * e.house.gamma / (
                                 1 + e.house.phi)
            budget[i] = budget[i - 1] - np.dot(prices[i - 1], q_exchange[i - 1, 0, 1:]) + np.sum(q_exchange[i, 1:, 0])
        return utility, budget
