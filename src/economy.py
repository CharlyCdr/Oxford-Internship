"""
The ``economy`` module
======================

This module declares the Economy class which encapsulates everything static in the model.
This class has the network attributes (both input-output and substitution) along with subsequent quantities
(equilibrium etc). It also enhirits firms and households attributes.
"""
import warnings

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy.optimize import leastsq

from firms import Firms
from household import Household
from network import create_net

warnings.simplefilter("ignore")


class Economy:
    """
    This class summarizes all the economy attributes.
    """

    @staticmethod
    def non_linear_eq_qnonzero(x, *p):
        """
        Function used for computation of equilibrium with non constant return to scale
        and general CES production function.
        :param x: guess for equilibrium
        :param p: tuple z, z_zeta, v, m_cal, zeta, q, theta, theta_bar, power
        :return: function's value at x
        """
        if len(x) % 2 != 0:
            raise ValueError("x must be of even length")

        # pylint: disable=unbalanced-tuple-unpacking
        u, w = np.split(x, 2)
        z_zeta, v, m_cal, q, exponent, kappa = p
        w_over_uq_p = np.power(np.divide(w, np.power(z_zeta, q) * np.power(u, q)), exponent)
        v1 = np.multiply(z_zeta, np.multiply(u, 1 - w_over_uq_p))
        m1 = np.dot(m_cal, u)
        m2 = u * np.dot(m_cal.T, w) - w * m1
        return np.concatenate((m1 - v1 - v, m2 + w * v - kappa))

    @staticmethod
    def non_linear_eq_qzero(x, *par):
        """
        Function used for computation of equilibrium with non constant return to scale
        and Leontieff production function.
        :param x: guess for equilibrium
        :param par: tuple z_zeta, v, m_cal, q power
        :return: function's value at x
        """

        if len(x) % 2 != 0:
            raise ValueError('x must be of even length')

        # pylint: disable=unbalanced-tuple-unpacking
        p, g = np.split(x, 2)
        z, v, m_cal, exponent, kappa = par
        v1 = np.multiply(z, np.multiply(p, 1 - np.power(g, exponent)))
        m1 = np.dot(m_cal, p)
        m2 = g * m1 - p * np.dot(m_cal.T, g)
        return np.concatenate((m1 - v1 - v, m2 - g * v + kappa))

    @staticmethod
    def adj_list(j):
        """
        :param j: Network adjacency matrix.
        :return: Representation of the network as an adjacency list.
        """
        return np.array([np.where(j[i] != 0)[0] for i in range(len(j))])

    def __init__(self, n, d, netstring, directed, j0, a0, q, b):

        # Network initialization
        self.n = n
        self.j = create_net(netstring, directed, n, d)
        self.j0 = j0
        self.j_a = None
        self.a0 = a0
        a = np.multiply(np.random.uniform(0, 1, (n, n)), self.j)
        self.a = np.array([(1 - a0[i]) * a[i] / np.sum(a[i]) for i in range(n)])
        self.a_a = None
        self.q = q
        self.zeta = 1 / (q + 1)
        self.b = b

        # Auxiliary network variables
        self.lamb = None
        self.a_a, self.j_a = None, None
        self.lamb_a = None
        self.m_cal = None
        self.v = None

        # Inheritances
        self.firms = None
        self.house = None

        # Equilibrium quantities
        self.p_eq = None
        self.g_eq = None
        self.mu_eq = None
        self.b_eq = None

    def init_house(self, labour, theta, gamma, phi):
        """
        Initialize a household object as instance of economy class. Cf household class.
        :param labour:
        :param theta:
        :param gamma:
        :param phi:
        :return:
        """
        self.house = Household(labour, theta, gamma, phi)

    def init_firms(self, z, sigma, alpha, alpha_p, beta, beta_p, w):
        """
        Initialize a firms object as instance of economy class. Cf firms class.
        :param z:
        :param sigma:
        :param alpha:
        :param alpha_p:
        :param beta:
        :param beta_p:
        :param w:
        :return:
        """
        self.firms = Firms(z, sigma, alpha, alpha_p, beta, beta_p, w)

    def set_house(self, house):
        """
        Sets a household object as instance of economy class.
        :param house:
        :return:
        """
        self.house = house

    def set_firms(self, firms):
        """
        Sets a firms object as instance of economy class.
        :param firms:
        :return:
        """
        self.firms = firms
        
    def update_firms_z(self, z):
        self.firms.update_z(z)
    
    def update_firms_sigma(self, sigma):
        self.firms.update_sigma(sigma)
        
    def update_firms_alpha(self, alpha):
        self.firms.update_alpha(alpha)
        
    def update_firms_alpha_p(self, alpha_p):
        self.firms.update_alpha_p(alpha_p)
        
    def update_firms_beta(self, beta):
        self.firms.update_beta(beta)
        
    def update_firms_beta_p(self, beta_p):
        self.firms.update_beta_p(beta_p)
    
    def update_firms_w(self, w):
        self.firms.update_w(w)
        
    def update_house_labour(self, labour):
        self.house.update_labour(labour)
        
    def update_house_theta(self, theta):
        self.house.update_theta(theta)
        
    def update_house_gamma(self, gamma):
        self.house.update_gamma(gamma)
        
    def update_house_phi(self, phi):
        self.house.update_phi(phi)

    def set_j(self, j):
        """
        Sets a particular input-output network.
        :param j: a n by n matrix.
        :return: side effect
        """
        if j.shape != (self.n, self.n):
            raise ValueError('Input-output network must be of size (%d, %d)' % (self.n, self.n))

        self.j = j

    def set_a(self, a):
        """
        Sets a particular input-output network.
        :param a: a n by n matrix.
        :return: side effect
        """
        if a.shape != (self.n, self.n):
            raise ValueError('Substitution network must be of size (%d, %d)' % (self.n, self.n))
        self.a = a

    def set_quantities(self):
        """
        Sets redundant economy quantities as class instances.
        :return: side effect
        """
        if self.q == 0:
            self.lamb = self.j
            self.a_a, self.j_a = np.hstack((np.array([self.a0]).T, self.a)), np.hstack((np.array([self.j0]).T, self.j))
            self.lamb_a = self.j_a
            self.m_cal = np.diag(self.firms.z) - self.lamb
            self.v = np.array(self.lamb_a[:, 0])
        elif self.q == np.inf:
            self.lamb = self.a
            self.a_a, self.j_a = np.hstack((np.array([self.a0]).T, self.a)), np.hstack((np.array([self.j0]).T, self.j))
            self.lamb_a = self.a_a
            self.m_cal = np.eye(self.n) - self.lamb
            self.v = np.array(self.lamb_a[:, 0])
        else:
            self.lamb = np.multiply(np.power(self.a, self.q * self.zeta), np.power(self.j, self.zeta))
            self.a_a, self.j_a = np.hstack((np.array([self.a0]).T, self.a)), np.hstack((np.array([self.j0]).T, self.j))
            self.lamb_a = np.multiply(np.power(self.a_a, self.q * self.zeta), np.power(self.j_a, self.zeta))
            self.m_cal = np.diag(np.power(self.firms.z, self.zeta)) - self.lamb
            self.v = np.array(self.lamb_a[:, 0])
        self.zeros_j_a = self.j_a != 0

    def get_eps_cal(self):
        """
        Computes the smallest eigenvalue of the economy matrix
        :return: smallest eigenvalue
        """
        return np.min(np.real(np.linalg.eigvals(self.m_cal)))

    def set_eps_cal(self, eps):
        """
        Modifies firms instance to set smallest eigenvalue of economy matrix to given epsilon.
        :param eps: a real number
        :return: side effect
        """
        min_eig = self.get_eps_cal()
        z_n, sigma, alpha, alpha_p, beta, beta_p, w = self.firms.z * np.power(1 + (eps - min_eig) /
                                                                              np.power(self.firms.z,
                                                                                       self.zeta),
                                                                              self.q + 1), \
                                                      self.firms.sigma, \
                                                      self.firms.alpha, \
                                                      self.firms.alpha_p, \
                                                      self.firms.beta, \
                                                      self.firms.beta_p, \
                                                      self.firms.w
        self.init_firms(z_n, sigma, alpha, alpha_p, beta, beta_p, w)
        self.set_quantities()

    def update_b(self, b):
        """
        Sets return to scale parameter
        :param b: return to scale
        :return: side effect
        """
        self.b = b
        self.compute_eq()

    def update_q(self, q):
        self.q = q
        self.zeta = 1 / (q + 1)
        self.set_quantities()

    def update_network(self, netstring, directed, d, n):
        self.j = create_net(netstring, directed, n, d)
        a = np.multiply(np.random.uniform(0, 1, (n, n)), self.j)
        self.a = np.array([(1 - self.a0[i]) * a[i] / np.sum(a[i]) for i in range(n)])
        self.set_quantities()

    def update_a0(self, a0):
        self.a0 = a0
        a = np.multiply(np.random.uniform(0, 1, (self.n, self.n)), self.j)
        self.a = np.array([(1 - self.a0[i]) * a[i] / np.sum(a[i]) for i in range(self.n)])
        self.set_quantities()

    def update_j0(self, j0):
        self.j0 = j0
        self.set_quantities()


    def production_function(self, Q):
        """
        CES production function
        :param Q: if n firms, (n,n+1) matrix of available labour and goods for production
        :return: productions of the n firms
        """
        if self.q == 0:
            return np.power(np.min(np.ma.masked_invalid(np.divide(Q, self.j_a)),
                                   axis=1),
                            self.b)
        elif self.q == np.inf:
            return np.power(np.prod(np.power(np.ma.masked_invalid(np.divide(Q, self.j_a)),
                                             self.a_a),
                                    axis=1),
                            self.b)
        else:
            return np.power(np.sum(np.ma.masked_invalid(self.a_a * np.power(self.j_a, 1. / self.q)
                                                        / np.power(Q, 1. / self.q)), axis=1),
                            - self.b * self.q)

    def compute_p_net(self, prices):
        """
        Compute the network prices
        :param prices: current rescaled prices
        :return: current wage-rescaled network prices
        """
        if self.q == np.inf:
            return np.sum(self.lamb_a, axis=1)
        else:
            return np.matmul(self.lamb_a, np.power(np.concatenate(([1], prices)), self.zeta))

    def compute_eq(self):
        """
        :return: compute the competitive equilibrium of the economy
        """
        # TODO: code COBB-DOUGLAS q=inf

        if self.q == np.inf:
            h = np.sum(self.a_a * np.log(np.ma.masked_invalid(np.divide(self.j_a, self.a_a))), axis=1)
            v = lstsq(np.eye(self.n)-self.a.T, self.house.kappa, rcond=10e-7)[0]
            log_p = lstsq(np.eye(self.n) / self.b - self.a,
                          - np.log(self.firms.z) / self.b + (1 - self.b) * np.log(v) / self.b + h, rcond=10e-7)[0]
            log_g = - np.log(self.firms.z) - log_p + np.log(v)
            self.p_eq, self.g_eq = np.exp(log_p), np.exp(log_g)
        else:
            if self.b != 1:
                if self.q == 0:
                    init_guess_peq = lstsq(self.m_cal, self.v, rcond=10e-7)[0]
                    init_guess_geq = lstsq(self.m_cal.T, np.divide(self.house.kappa, init_guess_peq), rcond=None)[0]

                    par = (self.firms.z,
                           self.v,
                           self.m_cal,
                           self.b - 1,
                           self.house.kappa)

                    pg = leastsq(lambda x: self.non_linear_eq_qzero(x, *par),
                                 np.concatenate((init_guess_peq, init_guess_geq)),

                                 )[0]
                    # pylint: disable=unbalanced-tuple-unpacking
                    self.p_eq, g = np.split(pg, 2)
                    self.g_eq = np.power(g, self.b)
                else:
                    init_guess_peq_zeta = lstsq(self.m_cal, self.v, rcond=None)[0]
                    init_guess_w = lstsq(self.m_cal.T,
                                         np.divide(self.house.kappa, init_guess_peq_zeta),
                                         rcond=None)[0]

                    par = (np.power(self.firms.z, self.zeta),
                           self.v,
                           self.m_cal,
                           self.q,
                           (self.b - 1) / (self.b * self.q + 1),
                           self.house.kappa
                           )

                    uw = leastsq(lambda x: self.non_linear_eq_qnonzero(x, *par),
                                 np.concatenate((init_guess_peq_zeta, init_guess_w)),
                                 )[0]

                    # pylint: disable=unbalanced-tuple-unpacking
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

    def save_eco(self, name):
        """
        Saves the economy in multi-indexed dataframe in hdf format.
        :param name: name of file.
        :return: None
        """
        first_index = np.concatenate((['Firms' for k in range(11)], ['Household' for k in range(4)]))
        second_index = np.concatenate(
            (['q', 'b', 'z', 'sigma', 'alpha', 'alpha_p', 'beta', 'beta_p', 'w', 'p_eq', 'g_eq'],
             ['l', 'theta', 'gamma', 'phi']))
        multi_index = [first_index, second_index]
        values = np.vstack((self.q * np.ones(self.n),
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

        df_eco = pd.DataFrame(values,
                              index=multi_index,
                              columns=np.arange(1, self.n + 1)
                              )
        df_eco.to_hdf(name + '/eco.h5', key='df', mode='w')
        np.save(name + '/network.npy', self.j_a)
        if self.q != 0:
            np.save(name + '/sub_network.npy', self.a_a)