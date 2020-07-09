import numba
import numpy as np
from scipy.optimize import fsolve


class Household(object):

    @staticmethod
    def fixed_point_mu(x, p):
        thetabar, vphi, phi, budget = p
        return (thetabar * vphi - np.power(x, 1. + 1 / phi)) / (budget * vphi) - x

    def __init__(self, labour, theta, gamma, phi):
        """
        Set the fundamental parameters of the household
        :param labour: quantity of labour for phi --> \infty
        :param theta: vector of preferences for the goods
        :param gamma: aversion to labour parameter
        :param phi: concavity parameter
        """
        self.l = labour
        self.theta = theta
        self.thetabar = np.sum(self.theta)
        self.theta = theta / self.thetabar
        self.gamma = gamma
        self.phi = phi
        self.v_phi = np.power(gamma, 1. / phi) / np.power(labour, 1 + 1. / phi)
        self.kappa = self.theta / np.power(self.thetabar * self.v_phi,
                                           self.phi / (1 + self.phi))

    def utility(self, consumption, working_hours):
        return np.sum(self.theta * np.log(consumption)) - self.gamma * np.power(np.sum(working_hours) / self.l,
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
            x0 = np.power(self.thetabar * self.v_phi, self.phi / (1 + self.phi)) / 2.
            mu = fsolve(self.fixed_point_mu, x0, args=(self.thetabar, self.v_phi, self.phi, budget))
        # At this stage, prices have to be rescaled by p0(t)

        return mu, self.theta / (mu * prices), np.power(mu, 1. / self.phi) / self.v_phi

    def compute_demand_cons_labour_supply_test(self, budget, prices, labour_prev, working_hours_prev):
        rat = np.clip(working_hours_prev / labour_prev, None, 1)
        mu = 0
        if self.phi == 1:
            mu = .5 * (np.sqrt(np.power(budget * self.v_phi, 2)
                               + 4 * rat * self.v_phi * self.thetabar)
                       - budget * self.v_phi) / rat
        return mu, self.theta / (mu * prices), np.power(mu * rat, 1. / self.phi) / self.v_phi

    @staticmethod
    @numba.jit
    def budget_constraint(budget, prices, offered_cons):
        b_vs_c = np.clip(budget / np.dot(offered_cons, prices), None, 1)
        cons_real = offered_cons * b_vs_c
        budget_res = budget - np.dot(cons_real, prices)
        return b_vs_c, cons_real, budget_res
