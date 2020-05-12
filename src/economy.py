import os, sys
sys.path.append('.')

import pandas as pd
import numpy as np
from numpy.linalg import lstsq
from scipy.optimize import fsolve, anderson, newton_krylov

from household import Household
from firms import Firms
from network import create_net
from exception import *


class Economy(object):

    @staticmethod
    def non_linear_eq_qnonzero(x, *p):
        """
        :param x:
        :param p: tuple z, z_zeta, v, m_cal, zeta, q, theta, theta_bar, power
        :return:
        """
        if len(x) % 2 != 0:
            raise ValueError("x must be of even length")

        u, w = np.split(x, 2)
        z, z_zeta, v, m_cal, zeta, q, theta, theta_bar, pow, kappa = p
        w_over_uq_p = np.power(np.divide(w, np.power(z_zeta, q) * np.power(u, q)), pow)
        v1 = np.multiply(z_zeta, np.multiply(u, 1 - w_over_uq_p))
        m1 = np.dot(m_cal, u)
        m2 = u * np.dot(m_cal.T, w) - w * m1
        return np.concatenate((m1 - v1 - v, m2 + w * v - kappa))

    @staticmethod
    def non_linear_eq_qzero(x, *par):
        """
        :param x:
        :param p: tuple z, z_zeta, v, m_cal, zeta, q, theta, theta_bar, power
        :return:
        """
        if len(x) % 2 != 0:
            raise ValueError("x must be of even length")

        p, g = np.split(x, 2)
        z, v, m_cal, pow, kappa, b = par
        v1 = np.multiply(z, np.multiply(p, 1 - np.power(g, pow)))
        m1 = np.dot(m_cal, p)
        m2 = g * m1 - p * np.dot(m_cal.T, g)
        return np.concatenate((m1 - v1 - v, m2 - g * v + kappa))

    @staticmethod
    def adj_list(j):
        l = []
        for i in range(len(j)):
            l.append(np.where(j[i] != 0)[0])
        return l

    def __init__(self, n, d, netstring, directed, j0, a0, q, b):
        self.n = n
        self.j = create_net(netstring, directed, n, d)
        self.j0 = j0
        self.j_a = None
        self.a0 = a0
        a = np.random.uniform(0, 1, (n, n))
        self.a = np.array([(1 - a0[i]) * a[i] / np.sum(a[i]) for i in range(n)])
        self.a_a = None
        self.q = q
        self.zeta = 1 / (q + 1)
        self.b = b

        self.p_eq = None
        self.g_eq = None

    def set_house(self, l, theta, gamma, phi):
        self.house = Household(l, theta, gamma, phi)

    def set_firms(self, z, sigma, alpha, alpha_p, beta, beta_p, w):
        self.firms = Firms(z, sigma, alpha, alpha_p, beta, beta_p, w)

    def save_eco(self, name):
        first_index = np.concatenate((['Firms' for k in range(11)], ['Household' for k in range(4)]))
        second_index = np.concatenate(
            (['q', 'b', 'z', 'sigma', 'alpha', 'alpha_p', 'beta', 'beta_p', 'w', 'p_eq', 'g_eq'],
             ['l', 'theta', 'gamma', 'phi']))
        multi_index = [first_index, second_index]
        vals = np.vstack((self.q * np.ones(self.n),
                          self.b * np.ones(self.n),
                          self.firms.z,
                          self.firms.sigma,
                          self.firms.alpha * np.ones(self.n),
                          self.firms.alpha_p * np.ones(self.n),
                          self.firms.beta * np.ones(self.n),
                          self.firms.beta_p * np.ones(self.n),
                          self.firms.w * np.ones(self.n),
                          self.p_eq,
                          self.g_eq,
                          self.house.l * np.ones(self.n),
                          self.house.theta,
                          self.house.gamma * np.ones(self.n),
                          self.house.phi * np.ones(self.n),
                          ))

        df_eco = pd.DataFrame(vals, index=multi_index, columns=np.arange(1, self.n + 1))
        df_eco.to_hdf(name + '_eco.h5', key='df', mode='w')
        np.save(name + '_network.npy', self.j_a)
        if self.q != 0:
            np.save(name + 'sub_network.npy', self.a_a)

    def set_quantities(self):
        if self.q == 0:
            self.lamb = self.j
            self.a_a, self.j_a = np.hstack((np.array([self.a0]).T, self.a)), np.hstack((np.array([self.j0]).T, self.j))
            self.lamb_a = self.j_a
            self.m_cal = np.diag(self.firms.z) - self.lamb
            self.v = np.array(self.lamb_a[:, 0])
        else:
            self.lamb = np.multiply(np.power(self.a, self.q * self.zeta), np.power(self.j, self.zeta))
            self.a_a, self.j_a = np.hstack((np.array([self.a0]).T, self.a)), np.hstack((np.array([self.j0]).T, self.j))
            self.lamb_a = np.multiply(np.power(self.a_a, self.q * self.zeta), np.power(self.j_a, self.zeta))
            self.m_cal = np.diag(np.power(self.firms.z, self.zeta)) - self.lamb
            self.v = np.array(self.lamb_a[:, 0])
        self.neigh = self.adj_list(self.j_a)

    def get_eps_cal(self):
        return np.min(np.real(np.linalg.eigvals(self.m_cal)))

    def production_function(self, Q):
        """
        CES production function
        :param Q: if n firms, (n,n+1) matrix of available labour and goods for production
        :return: productions of the n firms
        """
        if self.q == 0:
            return np.power(np.nanmin(np.divide(Q, self.j_a), axis=1), self.b)
        else:
            return np.power(np.nansum(self.a_a * np.power(self.j_a, 1. / self.q)
                                      / np.power(Q, 1. / self.q), axis=1),
                            - self.b * self.q)

    def compute_p_net(self, prices):
        '''
        Compute the network prices
        :param prices: current rescaled prices
        :return: current wage-rescaled network prices
        '''
        return np.matmul(self.lamb_a, np.power(np.concatenate(([1], prices)), self.zeta))

    def compute_eq(self):
        """
        :return: compute the equilibrium of the economy
        """

        if self.b != 1:
            if self.q == 0:
                init_guess_peq = lstsq(self.m_cal, self.v, rcond=10e-7)[0]
                init_guess_geq = lstsq(self.m_cal.T, np.divide(self.house.kappa, init_guess_peq), rcond=None)[0]

                par = (self.firms.z,
                       self.v,
                       self.m_cal,
                       self.b - 1,
                       self.house.kappa,
                       self.b)

                pg_init = anderson(lambda x: self.non_linear_eq_qzero(x, *par),
                              np.concatenate((init_guess_peq, init_guess_geq)),
                              M=50
                              )

                pg_int = anderson(lambda x: self.non_linear_eq_qzero(x, *par),
                              pg_init,
                              M=50
                              )

                pg = anderson(lambda x: self.non_linear_eq_qzero(x, *par),
                              pg_int,
                              M=50
                              )
                self.p_eq, g = np.split(pg, 2)
                self.g_eq = np.power(g, self.b)
            else:
                init_guess_peq_zeta = lstsq(self.m_cal, self.v, rcond=None)[0]
                init_guess_w = lstsq(self.m_cal.T,
                                       np.divide(self.house.kappa, init_guess_peq_zeta),
                                       rcond=None)[0]

                par = (self.firms.z,
                       np.power(self.firms.z, self.zeta),
                       self.v,
                       self.m_cal,
                       self.zeta,
                       self.q,
                       self.house.theta,
                       self.house.thetabar,
                       (self.b - 1) / (self.b * self.q + 1),
                       self.house.kappa
                       )

                uw = anderson(lambda x: self.non_linear_eq_qnonzero(x, *par),
                              np.concatenate((init_guess_peq_zeta, init_guess_w)),
                              M=50)

                u, w = np.split(uw, 2)
                uq = np.power(u, self.q)
                self.p_eq = np.power(u, 1. / self.zeta)
                self.g_eq = np.power(np.divide(w, np.power(self.firms.z, self.q * self.zeta) * uq),
                                     self.b / (self.zeta * (self.b * self.q + 1)))
        else:
            if self.q == 0:
                self.p_eq = lstsq(self.m_cal, self.v, rcond=10e-7)[0]
                self.g_eq = lstsq(self.m_cal.T, np.divide(self.house.kappa, self.p_eq), rcond=10e-7)[0]
            else:
                peq_zeta = lstsq(self.m_cal, self.v, rcond=None)[0]
                self.p_eq = np.power(peq_zeta, 1. / self.zeta)
                w = lstsq(self.m_cal.T, np.divide(self.house.kappa, peq_zeta), rcond=None)[0]
                self.g_eq = np.divide(w, np.power(self.firms.z, self.q * self.zeta) * np.power(peq_zeta, self.q))

        self.mu_eq = np.power(self.house.thetabar * self.house.v_phi, self.house.phi / (1 + self.house.phi))
        self.b_eq = self.house.thetabar / self.mu_eq
