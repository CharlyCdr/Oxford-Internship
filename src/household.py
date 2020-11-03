import numpy as np
from scipy.optimize import fsolve


class Household(object):

    def __init__(self, labour, theta, gamma, phi, w_p=None):
        """
        Set the fundamental parameters of the household
        :param labour: quantity of labour for phi --> \infty
        :param theta: vector of preferences for the goods
        :param gamma: aversion to labour parameter
        :param phi: concavity parameter
        """
        # Primary instances
        self.l = labour
        thetabar = np.sum(theta)
        self.theta = theta / thetabar
        self.thetabar = 1.
        self.gamma = gamma
        self.phi = phi
        self.w_p = w_p if w_p else 0

        # Secondary instances
        self.v_phi = np.power(self.gamma, 1. / self.phi) / np.power(self.l, 1 + 1. / self.phi)
        self.kappa = self.theta / np.power(self.thetabar * self.v_phi,
                                           self.phi / (1 + self.phi))

    def update_labour(self, labour):
        self.l = labour
        self.v_phi = np.power(self.gamma, 1. / self.phi) / np.power(labour, 1 + 1. / self.phi)
        self.kappa = self.theta / np.power(self.thetabar * self.v_phi,
                                           self.phi / (1 + self.phi))

    def update_theta(self, theta):
        thetabar = np.sum(theta)
        self.theta = theta / thetabar
        self.thetabar = 1.
        self.theta = theta
        self.v_phi = np.power(self.gamma, 1. / self.phi) / np.power(self.l, 1 + 1. / self.phi)
        self.kappa = self.theta / np.power(self.thetabar * self.v_phi,
                                           self.phi / (1 + self.phi))

    def update_gamma(self, gamma):
        self.gamma = gamma
        self.v_phi = np.power(self.gamma, 1. / self.phi) / np.power(self.l, 1 + 1. / self.phi)
        self.kappa = self.theta / np.power(self.thetabar * self.v_phi,
                                           self.phi / (1 + self.phi))

    def update_phi(self, phi):
        self.phi = phi
        self.v_phi = np.power(self.gamma, 1. / self.phi) / np.power(self.l, 1 + 1. / self.phi)
        self.kappa = self.theta / np.power(self.thetabar * self.v_phi,
                                           self.phi / (1 + self.phi))

    def utility(self, consumption, working_hours):
        return np.sum(self.theta * np.log(consumption)) - self.gamma * np.power(working_hours.sum() / self.l,
                                                                                1. + self.phi) / (
                       1. + self.phi)

    def compute_demand_cons_labour_supply(self, budget, prices, supply, demand, lda):
        theta = self.theta * np.exp(-self.w_p * (supply - demand)/(supply + demand))
        b_new = budget + (1 - lda) * demand

        if self.phi == 1:
            mu = .5 * (np.sqrt(np.power(b_new * self.v_phi, 2)
                               + 4 * self.v_phi * np.sum(theta) * lda ** 2)
                       - b_new * self.v_phi) / lda ** 2
        elif self.phi == np.inf:
            mu = np.sum(theta) / (self.l + budget)
            raise Exception('Not coded yet')
        else:
            raise Exception('Not coded yet')
            # x0 = np.power(self.thetabar * self.v_phi, self.phi / (1 + self.phi)) / 2.
            # mu = fsolve(self.fixed_point_mu, x0, args=(self.thetabar, self.v_phi, self.phi, budget))

        return theta / (mu * prices), np.power(mu * lda, 1. / self.phi) / self.v_phi

    def budget_constraint(self, budget, prices, offered_cons):
            b_vs_c = np.minimum(budget / np.dot(offered_cons, prices), 1)
            cons_real = offered_cons * b_vs_c
            budget_res = budget - np.dot(cons_real, prices)
            return b_vs_c, cons_real, budget_res

    def fixed_point_mu(self, x, p):
        thetabar, vphi, phi, budget = p
        return (thetabar * vphi - np.power(x, 1. + 1 / phi)) / (budget * vphi) - x