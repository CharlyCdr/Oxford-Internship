"""
The ``firms`` module
======================

This module declares the Firms class which model n firms.
The attributes of this class are all the fixed parameters defining firms
z : productivity factors
sigma;: depreciation of stocks parameter
alpha, alpha_p, beta, beta_p, w : inverse time-scales for feed-backs.
The methods of this class encode the way firms update the varying quantities such as prices, productions etc...
"""

import numpy as np
from numba import njit, float64


@njit
def clip_max(a, limit):
    a_copy = a
    for i in range(a.shape[0]):
        if a[i] > limit:
            a_copy[i] = limit
        else:
            a_copy[i] = a[i]
    return a_copy


@njit
def clip_min(a, limit):
    a_copy = a
    for i in range(a.shape[0]):
        if a[i] < limit:
            a_copy[i] = limit
        else:
            a_copy[i] = a[i]
    return a_copy


spec = [
    ('z', float64[:]),
    ('sigma', float64[:]),
    ('alpha', float64),
    ('alpha_p', float64),
    ('beta', float64),
    ('beta_p', float64),
    ('w', float64)
]


class Firms:
    def __init__(self, z, sigma, alpha, alpha_p, beta, beta_p, omega):

        if (z < 0).any():
            raise Exception("Productivity factors must be positive.")
        if (np.array([alpha, alpha_p, beta, beta_p]) < 0).any():
            raise Exception("Inverse timescales must be positive.")
        if (sigma < 0).any():
            raise Exception("Depreciation of stocks must be positive.")

        # Production function parameters
        self.z = z
        self.sigma = sigma
        self.alpha = alpha
        self.alpha_p = alpha_p
        self.beta = beta
        self.beta_p = beta_p
        self.omega = omega

    def update_z(self, z):
        self.z = z

    def update_sigma(self, sigma):
        self.sigma = sigma

    def update_alpha(self, alpha):
        self.alpha = alpha

    def update_alpha_p(self, alpha_p):
        self.alpha_p = alpha_p

    def update_beta(self, beta):
        self.beta = beta

    def update_beta_p(self, beta_p):
        self.beta_p = beta_p

    def update_w(self, omega):
        self.omega = omega

    def update_prices(self, prices, profits, balance, cashflow, tradeflow, step_s):
        """
        Updates prices according to observed profits and balances
        :param prices: current wage-rescaled prices
        :param profits: current wages-rescaled profits
        :param balance: current balance
        :param cashflow: current wages-rescaled gain + losses
        :param tradeflow: current supply + demand
        :return:
        """
        return prices * np.exp(- step_s * (self.alpha_p * profits / cashflow + self.alpha * balance[1:] / tradeflow[1:]))

    def update_wages(self, labour_balance, total_labour, step_s):
        """
        Updates wages according to the observed tensions in the labour market
        :param labour_balance: labour supply - labour demand
        :param total_labour: labour supply + labour demand
        :return: Updated wage
        """
        return np.exp(- self.omega * step_s * labour_balance / total_labour)

    def compute_targets(self, prices, Q_demand_prev, supply, prods, step_s):
        """
        Computes the production target based on profit and balance forecasts.
        :param prices: current rescaled prices
        :param Q_demand_prev: (n+1, n+1) matrix of goods and labour demands of previous period along with consumption
                                demands
        :param supply: current supply
        :param prods: current production levels
        :return: Production targets for the next period
        """
        est_profits, est_balance, est_cashflow, est_tradeflow = self.compute_forecasts(prices, Q_demand_prev, supply)
        return prods * np.exp(step_s * (self.beta * est_profits / est_cashflow
                              - self.beta_p * est_balance[1:] / est_tradeflow[1:]))

    def compute_forecasts(self, prices, Q_demand_prev, supply):
        """
        Computes the expected profits and balances assuming same demands as previous time
        :param prices: current wage-rescaled prices
        :param Q_demand_prev: (n+1, n+1) matrix of goods and labour demands of previous period along with consumption
                                demands
        :param supply: current supply
        :return: Forecasts of gains - losses, supply - demand, gains + losses, supply + demand
        """

        exp_gain = prices * np.sum(Q_demand_prev[:, 1:], axis=0)
        exp_losses = np.matmul(Q_demand_prev[1:, :], np.concatenate((np.array([1]), prices)))
        exp_supply = supply
        exp_demand = np.sum(Q_demand_prev, axis=0)
        return exp_gain - exp_losses, exp_supply - exp_demand, exp_gain + exp_losses, exp_supply + exp_demand

    def compute_optimal_quantities(self, targets, prices, e):
        """
        Computes
        :param e: economy class
        :param targets: production targets for the next period
        :param prices_net: current wage-rescaled aggregated network prices
        :param prices: current wages-rescaled aggregated network prices
        :return: (n, n+1) matrix of labour/goods demands
        """
        if e.q == 0:
            demanded_products_labor = np.matmul(np.diag(np.power(targets, 1. / e.b)),
                                                e.lamb_a)
        elif e.q == np.inf:
            prices_net_aux = np.array([
                np.prod(np.power(e.j_a[i, :] * np.concatenate((np.array([1]), prices)) /
                                 e.a_a[i, :], e.a_a[i, :])[e.zeros_j_a[i, :]])
                for i in range(e.n)
            ])
            demanded_products_labor = np.multiply(e.a_a,
                                                  np.outer(np.multiply(prices_net_aux,
                                                                       np.power(targets, 1. / e.b)),
                                                           np.concatenate((np.array([1]), 1. / prices))
                                                           ))
        else:
            prices_net = np.matmul(e.lamb_a, np.power(np.concatenate(([1], prices)), e.zeta))
            demanded_products_labor = np.multiply(e.lamb_a,
                                                  np.outer(np.multiply(np.power(prices_net, e.q),
                                                                       np.power(targets, 1. / e.b)),
                                                           np.power(np.concatenate((np.array([1]), prices)),
                                                                    - e.q / (1 + e.q))))
        return demanded_products_labor

    def compute_profits_balance(self, prices, Q_real, supply, demand):
        """
        Compute the real profits and balances of firms
        :param prices: current wage-rescaled prices
        :param Q_real: (n+1, n+1) matrix of exchanged goods, labour and consumption
        :param supply: current supply
        :param demand: current demand
        :return: Real wage-rescaled values of gains - losses, supply - demand, gains + losses, supply + demand
        """
        gain = prices * np.sum(Q_real[:, 1:], axis=0)
        losses = np.matmul(Q_real[1:, :], np.concatenate((np.array([1]), prices)))

        return gain - losses, supply - demand, gain + losses, supply + demand
