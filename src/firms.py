# network-economy is a simulation program for the Network Economy ABM desbribed in (TODO)
# Copyright (C) 2020 Théo Dessertaine
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
The ``firms`` module
======================

This module declares the Firms class which model n firms.
The attributes of this class are all the fixed parameters defining firms.
The methods of this class encode the way firms update the varying quantities such as prices, productions etc...
"""

import numpy as np


class Firms:
    def __init__(self, z, sigma, alpha, alpha_p, beta, beta_p, omega):

        if (z < 0).any():
            raise Exception("Productivity factors must be positive.")
        if (np.array([alpha, alpha_p, beta, beta_p]) < 0).any():
            raise Exception("Inverse timescales must be positive.")
        if (sigma < 0).any():
            raise Exception("Depreciation of stocks must be positive.")

        # Production function parameters
        self.z = z  # Productivity factors
        self.sigma = sigma  # Depreciation of stocks parameters
        self.alpha = alpha  # Log-elasticity of prices' growth rates against surplus
        self.alpha_p = alpha_p  # Log-elasticity of prices' growth rates against profits
        self.beta = beta  # Log-elasticity of productions' growth rates against profits
        self.beta_p = beta_p  # Log-elasticity of productions' growth rates against surplus
        self.omega = omega  # Log-elasticity of wages' growth rates against labor-market tensions

    # Setters for class instances

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
        Updates prices according to observed profits and balances.
        :param prices: current wage-rescaled prices,
        :param profits: current wages-rescaled profits,
        :param balance: current balance,
        :param cashflow: current wages-rescaled gain + losses,
        :param tradeflow: current supply + demand,
        :param step_s: size of time step,
        :return: Updated prices for the next period.
        """
        return prices * np.exp(- 2 * step_s * (self.alpha_p * profits / cashflow +
                                           self.alpha * balance[1:] / tradeflow[1:]))

    def update_wages(self, labour_balance, total_labour, step_s):
        """
        Updates wages according to the observed tensions in the labour market.
        :param labour_balance: labour supply - labour demand,
        :param total_labour: labour supply + labour demand,
        :param step_s: size of time-step,
        :return: Updated wage for the next period.
        """
        return np.exp(- 2 * self.omega * step_s * (labour_balance / total_labour))

    def compute_targets(self, prices, q_forecast, supply, prods, step_s):
        """
        Computes the production target based on profit and balance forecasts.
        :param prices: current rescaled prices,
        :param q_forecast: forecast exchanged quantities,
        :param supply: current supply,
        :param prods: current production levels,
        :param step_s: size of time-step,
        :return: Production targets for the next period.
        """
        est_profits, est_balance, est_cashflow, est_tradeflow = self.compute_forecasts(prices, q_forecast, supply)
        return prods * np.exp(2 * step_s * (self.beta * est_profits / est_cashflow
                              - self.beta_p * est_balance[1:] / est_tradeflow[1:]))

    @staticmethod
    def compute_profits_balance(prices, q_exchange, supply, demand):
        """
        Compute the real profits and balances of firms.
        :param prices: current wage-rescaled prices,
        :param q_exchange: matrix of exchanged goods, labor and consumptions,
        :param supply: current supply,
        :param demand: current demand,
        :return: Realized wage-rescaled values of profits, balance, cash-flow, trade-flow.
        """
        gain = prices * np.sum(q_exchange[:, 1:], axis=0)
        losses = np.matmul(q_exchange[1:, :], np.concatenate((np.array([1]), prices)))

        return gain - losses, supply - demand, gain + losses, supply + demand

    @staticmethod
    def compute_optimal_quantities(targets, prices, e):
        """
        Computes minimizing-costs quantities given different production functions and production target.
        :param e: economy class,
        :param targets: production targets for the next period,
        :param prices: current wages-rescaled prices,
        :return: Matrix of optimal goods/labor quantities.
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

    @staticmethod
    def compute_forecasts(prices, q_forecast, supply):
        """
        Computes the expected profits and balances assuming same demands as previous time.
        :param prices: current wage-rescaled prices,
        :param q_forecast: forecast exchanged quantities,
        :param supply: current supply,
        :return: Forecast of profits, balance, cash-flow and trade-flow.
        """
        exp_gain = prices * np.sum(q_forecast[:, 1:], axis=0)
        exp_losses = np.matmul(q_forecast[1:, :], np.concatenate((np.array([1]), prices)))
        exp_supply = supply
        exp_demand = np.sum(q_forecast, axis=0)
        return exp_gain - exp_losses, exp_supply - exp_demand, exp_gain + exp_losses, exp_supply + exp_demand
