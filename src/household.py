import numpy as np
from scipy.optimize import fsolve


class Household(object):

    def __init__(self, labour, theta, gamma, phi, omega_p=None):
        """
        Set the fundamental parameters of the household
        :param labour: quantity of labour for phi --> \infty
        :param theta: vector of preferences for the goods
        :param gamma: aversion to labour parameter
        :param phi: concavity parameter
        """
        # Primary instances
        self.l = labour
        self.theta = theta
        self.thetabar = np.sum(theta)
        self.gamma = gamma
        self.phi = phi
        self.omega_p = omega_p if omega_p else 0

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

    def update_w_p(self, omega_p):
        self.omega_p = omega_p

    def utility(self, consumption, working_hours):
        return np.sum(self.theta * np.log(consumption)) - self.gamma * np.power(working_hours.sum() / self.l,
                                                                                1. + self.phi) / (
                       1. + self.phi)

    def compute_demand_cons_labour_supply(self, budget, prices, supply, demand, step_s):
        theta = self.theta * np.exp(-self.omega_p * step_s * (supply - demand) / (supply + demand))

        if self.phi == 1:
            mu = .5 * (np.sqrt(np.power(budget * self.v_phi, 2)
                               + 4 * self.v_phi * np.sum(theta))
                       - budget * self.v_phi)
        elif self.phi == np.inf:
            mu = np.sum(theta) / (self.l + budget)
        else:
            raise Exception('Not coded yet')
            # x0 = np.power(self.thetabar * self.v_phi, self.phi / (1 + self.phi)) / 2.
            # mu = fsolve(self.fixed_point_mu, x0, args=(self.thetabar, self.v_phi, self.phi, budget))

        return theta / (mu * prices), np.power(mu, 1. / self.phi) / self.v_phi

    def fixed_point_mu(self, x, p):
        thetabar, vphi, phi, budget = p
        return (thetabar * vphi - np.power(x, 1. + 1 / phi)) / (budget * vphi) - x