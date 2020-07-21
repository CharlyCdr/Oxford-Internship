import numpy as np
from scipy.optimize import fsolve
from numba import jitclass
from numba import float64

spec = [
    ('l', float64),
    ('theta', float64[:]),
    ('thetabar', float64),
    ('gamma', float64),
    ('phi', float64),
    ('v_phi', float64),
    ('kappa', float64[:])
]

class Household(object):

    def __init__(self, labour, theta, gamma, phi):
        """
        Set the fundamental parameters of the household
        :param labour: quantity of labour for phi --> \infty
        :param theta: vector of preferences for the goods
        :param gamma: aversion to labour parameter
        :param phi: concavity parameter
        """
        self.l = labour
        thetabar = np.sum(theta)
        self.theta = theta / thetabar
        self.thetabar = 1.
        self.gamma = gamma
        self.phi = phi
        self.v_phi = np.power(gamma, 1. / phi) / np.power(labour, 1 + 1. / phi)
        self.kappa = self.theta / np.power(self.thetabar * self.v_phi,
                                           self.phi / (1 + self.phi))

    def utility(self, consumption, working_hours):
        return np.sum(self.theta * np.log(consumption)) - self.gamma * np.power(working_hours.sum() / self.l,
                                                                                1. + self.phi) / (
                       1. + self.phi)

    def compute_demand_cons_labour_supply(self, budget, prices):
        if self.phi == 1:
            mu = .5 * (np.sqrt(np.power(budget * self.v_phi, 2)
                               + 4 * self.v_phi * self.thetabar)
                       - budget * self.v_phi)
        elif self.phi == np.inf:
            mu = self.thetabar / (self.l + budget)
        else:
            raise Exception('Not coded yet')
            # x0 = np.power(self.thetabar * self.v_phi, self.phi / (1 + self.phi)) / 2.
            # mu = fsolve(self.fixed_point_mu, x0, args=(self.thetabar, self.v_phi, self.phi, budget))

        return mu, self.theta / (mu * prices), np.power(mu, 1. / self.phi) / self.v_phi

    def budget_constraint(self, budget, prices, offered_cons):
        b_vs_c = np.minimum(budget / np.dot(offered_cons, prices), 1)
        cons_real = offered_cons * b_vs_c
        budget_res = budget - np.dot(cons_real, prices)
        return b_vs_c, cons_real, budget_res

    def fixed_point_mu(self, x, p):
        thetabar, vphi, phi, budget = p
        return (thetabar * vphi - np.power(x, 1. + 1 / phi)) / (budget * vphi) - x