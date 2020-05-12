from exception import *

import numpy as np


class Firms(object):
    def __init__(self, z, sigma, alpha, alpha_p, beta, beta_p, w):

        if (z < 0).any():
            raise InputError("Productivity factors must be positive")
        if (np.array([alpha, alpha_p, beta, beta_p]) < 0).any():
            raise InputError("Inverse timescales must be positive")
        if (sigma > 1).any() or (sigma < 0).any():
            raise InputError("Depreciation of stocks must be between 0 and 1")

        # Production function parameters
        self.z = z
        self.sigma = sigma
        self.alpha = alpha
        self.alpha_p = alpha_p
        self.beta = beta
        self.beta_p = beta_p
        self.w = w

    def update_prices(self, prices, profits, balance, cashflow, tradeflow):
        """
        Updates prices according to observed profits and balances
        :param prices: current wage-rescaled prices
        :param profits: current wages-rescaled profits
        :param balance: current balance
        :param cashflow: current wages-rescaled gain + losses
        :param tradeflow: current supply + demand
        :return:
        """
        return prices * (1 - self.alpha_p * (profits / cashflow) - self.alpha * (balance[1:] / tradeflow[1:]))

    def update_stocks(self, supply, sales):
        """
        Updates stocks
        :param supply: current supply
        :param sales: current sales
        :return: Depreciated unsold goods
        """
        return (1 - self.sigma) * np.clip(supply - sales, 0, None)

    def update_wages(self, labour_balance, total_labour):
        """
        Updates wages according to the observed tensions in the labour market
        :param labour_balance: labour supply - labour demand
        :param total_labour: labour supply + labour demand
        :return: Updated wage
        """
        return 1 - self.w * labour_balance / total_labour

    @staticmethod
    def compute_forecasts(prices, Q_demand_prev, supply):
        """
        Computes the expected profits and balances assuming same demands as previous time
        :param prices: current wage-rescaled prices
        :param Q_demand_prev: (n+1, n+1) matrix of goods and labour demands of previous period along with consumption demands
        :param supply: current supply
        :return: Forecasts of gains - losses, supply - demand, gains + losses, supply + demand
        """

        exp_gain = np.multiply(prices, np.sum(Q_demand_prev[:, 1:], axis=0))
        exp_losses = np.matmul(Q_demand_prev[1:, :], np.concatenate(([1], prices)))
        exp_supply = supply
        exp_demand = np.sum(Q_demand_prev, axis=0)
        return exp_gain - exp_losses, exp_supply - exp_demand, exp_gain + exp_losses, exp_supply + exp_demand

    def compute_targets(self, prices, Q_demand_prev, supply, prods):
        """
        Computes the production target based on profit and balance forecasts.
        :param prices: current rescaled prices
        :param Q_demand_prev: (n+1, n+1) matrix of goods and labour demands of previous period along with consumption demands
        :param supply: current supply
        :param prods: current production levels
        :return: Production targets for the next period
        """
        est_profits, est_balance, est_cashflow, est_tradeflow = self.compute_forecasts(prices, Q_demand_prev, supply)
        return prods * (
                    1 + self.beta * (est_profits / est_cashflow) - self.beta_p * (est_balance[1:] / est_tradeflow[1:]))

    @staticmethod
    def compute_demands_firms(targets, prices, prices_net, q, b, lamb_a):
        """
        Computes
        :param targets: production targets for the next period
        :param prices_net: current wage-rescaled aggregated network prices
        :param prices: current wages-rescaled aggregated network prices
        :param q: CES interpolator
        :param b: Return to scale parameter
        :param lamb_a: Aggregated network-subsitution matrix
        :return: (n, n+1) matrix of labour/goods demands
        """
        if q == 0:
            demanded_products_labor = np.matmul(np.diag(np.power(targets, 1. / b)),
                                                lamb_a)
        else:
            demanded_products_labor = np.multiply(lamb_a,
                                                  np.outer(np.multiply(np.power(prices_net, q),
                                                                       np.power(targets, 1. / b)),
                                                           np.power(np.concatenate(([1], prices)),
                                                                    - q / (1 + q))))
        return demanded_products_labor

    @staticmethod
    def compute_profits_balance(prices, Q, supply, demand):
        """
        Compute the real profits and balances of firms
        :param prices: current wage-rescaled prices
        :param Q: (n+1, n+1) matrix of exchanged goods, labour and consumption
        :param supply: current supply
        :param demand: current demand
        :return: Real wage-rescaled values of gains - losses, supply - demand, gains + losses, supply + demand
        """
        gain = np.multiply(prices, np.sum(Q[:, 1:], axis=0))
        losses = np.matmul(Q[1:, :], np.concatenate(([1], prices)))

        return gain - losses, supply - demand, gain + losses, supply + demand

