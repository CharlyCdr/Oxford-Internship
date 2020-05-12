import os, sys
import numpy as np
import scipy.linalg as scla
import scipy.optimize as scop
import scipy.integrate as sci
import networkx as nx



def eq_root_function(x, *par):
    z, m, b, q, v, mu, theta = par

    u, g = np.split(x, 2)
    n = len(u)
    u_arg = np.array(np.dot(m, u) - v + z * u * (1 - 1. / np.power(g, (1-b)/(b*(1+q)))).reshape(n))[0]
    g_arg = np.array(np.dot(m.T * np.diag(np.power(u, q)), np.power(g, (b*q+1)/(b*q+b))) - mu*theta/u + \
                z * np.power(u, q) * g * (1 - 1. / np.power(g, (1-b)/(b*(1+q)))).reshape(n))[0]
    return np.concatenate((u_arg, g_arg))

class Error(Exception):
    pass


class NoTypeError(Error):
    pass


class InputError(Error):
    pass


class NotCodedError(Error):
    pass



class Economy(object):

    def __init__(self, n, d, z, J, V, theta):
        self.n = n
        self.d = d
        self.z = z
        self.J = J
        self.V = V
        self.theta = theta
        self.A = None
        self.M = None
        self.directed = None
        self.p_eq = None
        self.lda_eq = None
        self.Sd = None
        self.Sc = None

    def set_M(self,A):
        self.A = A
        self.M = np.diag(self.z) - np.multiply(self.J, A)

    def set_M_eps(self, eps, A):
        self.A = A
        M = np.diag(self.z) - np.multiply(self.J, self.A)
        w = np.real(scla.eigvals(M))
        self.z += eps - np.min(w)
        self.M = np.diag(self.z) - np.multiply(self.J, self.A)

    def gen_M(self, eps, directed=False, ntype=None):
        self.directed = directed
        if not directed:
            g = nx.random_regular_graph(self.d, self.n)
            self.A = nx.convert_matrix.to_numpy_matrix(g)
        else:
            if ntype == 'regular':
                A = np.zeros((self.n, self.n))
                ind = np.random.choice(np.arange(1, self.n), self.d, replace=False)
                A[0, ind] = 1
                for k in range(1, self.n):
                    sums = np.sum(A, axis=0)
                    m = np.where(np.min(sums) == sums)[0]
                    m = m[m != k]
                    if len(m) < self.d:
                        r = len(m)
                        it = r
                        it_aux = 1
                        while it < self.d:
                            maux = np.where(np.min(sums) + it_aux == sums)[0]
                            maux = maux[maux != k]
                            if len(maux) < self.d - it:
                                m = np.append(m, maux)
                            else:
                                aux = np.random.choice(maux, self.d - it, replace=False)
                                m = np.append(m, aux)
                            it += min(len(maux), self.d - it)
                            it_aux += 1
                        ind = m
                    else:
                        ind = np.random.choice(m, self.d, replace=False)
                    A[k, ind] = 1
                self.A = A
            elif ntype == 'm_regular':
                A1 = nx.convert_matrix.to_numpy_matrix(nx.random_regular_graph(self.d, self.n))
                A2 = nx.convert_matrix.to_numpy_matrix(nx.random_regular_graph(self.d, self.n))
                A = np.triu(A1) + np.tril(A2)
                self.A = A
            else:
                raise NoTypeError('You must provide a type for the adjacency matrix')

        M = np.diag(self.z) - np.multiply(self.J, self.A)
        print(self.z)
        w = np.real(scla.eigvals(M))
        print(eps-np.min(w.real))
        self.z += eps - np.min(w.real)
        print(self.z)
        self.M = np.diag(self.z) - np.multiply(self.J, self.A)

    def compute_p(self):
        # Compute the equilibrium prices by minimizing ||Mp_{eq}-V||_{2}
        self.p_eq = scla.lstsq(self.M, self.V)[0]

    def compute_lda(self):
        """Compute the equilibrium productions by minimizing ||M^{t}l_{eq}-mu*\frac{theta}{p_{eq}}||_{2}"""
        mu = 1. / (self.n * np.nanmean(self.theta))
        self.lda_eq = scla.lstsq(np.transpose(self.M), mu * np.divide(self.theta, self.p_eq))[0]

    def set_Sc(self, alpha, beta):
        mu = 1. / (self.n * np.nanmean(self.theta))
        s11 = -alpha * mu * np.diag(self.theta / (self.z * self.lda_eq * self.p_eq))
        s12 = -alpha * np.dot(np.diag(self.p_eq / (self.z * self.lda_eq)), np.transpose(self.M))
        s21 = beta * np.dot(np.diag(self.lda_eq / (self.z * self.p_eq)), self.M) - beta * mu * np.diag(
            self.theta / (self.z * self.p_eq * self.p_eq))
        s22 = -beta * np.dot(np.diag(1 / self.z), np.transpose(self.M))
        self.Sc = np.block([[s11, s12],
                            [s21, s22]])

    def set_Sd(self, alpha, beta, q):
        mu = 1. / (self.n * np.nanmean(self.theta))
        a11 = np.eye(self.n) - alpha * mu * np.diag(self.theta / (self.z * self.lda_eq * self.p_eq))
        a12 = alpha * (1 + q) * np.dot(np.diag(self.p_eq / (self.z * self.lda_eq)),
                                       np.transpose(np.multiply(self.J, self.A)))
        a21 = beta * np.diag(self.lda_eq / self.p_eq) - beta * mu * np.diag(
            self.theta / (self.z * self.p_eq * self.p_eq))
        a22 = (1 - beta) * np.eye(self.n) + beta * (1+q) * np.dot(np.diag(1 / self.z),
                                                              np.transpose(np.multiply(self.J, self.A)))

        b11 = np.zeros((self.n, self.n))
        b12 = -alpha * np.diag(self.p_eq / self.lda_eq) - alpha * q * np.dot(
            np.diag(self.p_eq / (self.z * self.lda_eq)), np.transpose(np.multiply(self.J, self.A)))
        b21 = -beta * np.dot(np.diag(self.lda_eq / (self.z * self.p_eq)), np.multiply(self.J, self.A))
        b22 = -beta * q * np.dot(np.diag(1 / self.z), np.transpose(np.multiply(self.J, self.A)))

        a = np.block([[a11, a12],
                      [a21, a22]])
        b = np.block([[b11, b12],
                      [b21, b22]])

        self.Sd = np.block([[a, b],
                            [np.eye(2 * self.n), np.zeros((2 * self.n, 2 * self.n))]])

    def get_discrete_spectrum(self):
        return np.linalg.eig(self.Sd)

    def get_discrete_eigval(self):
        return np.linalg.eigvals(self.Sd)

    def get_discrete_eigvec(self):
        return np.linalg.eig(self.Sd)[1]

    def get_cont_spectrum(self):
        return np.linalg.eig(self.Sc)

    def get_cont_eigval(self):
        return np.linalg.eigvals(self.Sc)

    def get_cont_eigvec(self):
        return np.linalg.eig(self.Sc)[1]

    def critical_cont_spectrum(self, alpha, beta):
        "Coded for homogeneous z and undirected J"
        w = np.diag(np.min(self.z)-self.z) + np.linalg.eigh(self.A)[0]
        wm = np.max(w)
        w = wm - w
        if 4 * alpha <= beta:
            m1 = np.sqrt(alpha * beta) / (wm * (.5 * np.sqrt(beta / alpha) - np.sqrt(-1 + beta / (4*alpha))))
            m2 = np.sqrt(alpha * beta) / (wm * (.5 * np.sqrt(beta / alpha) + np.sqrt(-1 + beta / (4 * alpha))))
        else:
            m1 = np.sqrt(alpha * beta) / (wm * (.5 * np.sqrt(beta / alpha) - 1j * np.sqrt( 1 - beta / (4 * alpha))))
            m2 = np.sqrt(alpha * beta) / (wm * (.5 * np.sqrt(beta / alpha) + 1j * np.sqrt( 1 - beta / (4 * alpha))))
        return np.concatenate((- m1 * w, - m2 * w))

    def critical_discrete_spectrum_nullalpha(e, beta, q):
        "Coded for homogeneous z and undirected J"
        w = np.linalg.eigvals(e.A)
        rho_n = np.max(w)
        w = rho_n - w
        eig_th = np.array([], dtype=np.complex128)
        for omega in w:
            if omega == 0:
                eig_th = np.append(eig_th, [0, 0, 1, 1])
            else:
                eig_th = np.append(eig_th, [0, 1])
                Delta = 1 - 2 * beta * (q + omega * (1 - q) / rho_n) + beta ** 2 * (q - omega * (1 + q) / rho_n) ** 2
                delta = 16 * q * omega * (rho_n - omega) / rho_n ** 2
                if q * (rho_n - omega) < 0:
                    eig_th = np.append(eig_th, [.5 * (1 + beta * (q - omega * (1 + q) / rho_n) + np.sqrt(Delta)),
                                                .5 * (1 + beta * (q - omega * (1 + q) / rho_n) - np.sqrt(Delta))])
                elif q * (rho_n - omega) == 0:
                    beta_0 = (q + omega * (1 - q) / rho_n) / ((q - omega * (1 + q) / rho_n) ** 2)
                    if beta == beta_0:
                        eig_th = np.append(eig_th, [.5 * (1 + beta * (q - omega * (1 + q) / rho_n)),
                                                    .5 * (1 + beta * (q - omega * (1 + q) / rho_n))])
                    else:
                        eig_th = np.append(eig_th, [.5 * (1 + beta * (q - omega * (1 + q) / rho_n) + np.abs(
                            q - omega * (1 + q) / rho_n) * np.abs(beta - beta_0)),
                                                    .5 * (1 + beta * (q - omega * (1 + q) / rho_n) - np.abs(
                                                        q - omega * (1 + q) / rho_n) * np.abs(beta - beta_0))])
                else:
                    beta_p, beta_m = (q + omega * (1 - q) / rho_n + 2 * np.sqrt(delta / 16.)) / (
                            q - omega * (1 + q) / rho_n) ** 2, (
                                             q + omega * (1 - q) / rho_n - 2 * np.sqrt(delta / 16.)) / (
                                             q - omega * (1 + q) / rho_n) ** 2
                    if beta > min(beta_p, beta_m) and beta < max(beta_p, beta_m):
                        eig_th = np.append(eig_th,
                                           [.5 * (1 + beta * (q - omega * (1 + q) / rho_n) + 1j * np.sqrt(-Delta)),
                                            .5 * (1 + beta * (q - omega * (1 + q) / rho_n) - 1j * np.sqrt(-Delta))])
                    elif beta == min(beta_p, beta_m) and beta == max(beta_p, beta_m):
                        eig_th = np.append(eig_th,
                                           [.5 * (1 + beta * (q - omega * (1 + q) / rho_n)),
                                            .5 * (1 + beta * (q - omega * (1 + q) / rho_n))])
                    else:
                        eig_th = np.append(eig_th,
                                           [.5 * (1 + beta * (q - omega * (1 + q) / rho_n) + np.sqrt(Delta)),
                                            .5 * (1 + beta * (q - omega * (1 + q) / rho_n) - np.sqrt(Delta))])

        return eig_th

class Dynamics(object):

    def __init__(self, n):
        self.n = n
        self.eco = None
        self.dis_dyn = None
        self.con_dyn = None
        self.dis_lin_dyn = None
        self.con_lin_dyn = None
        self.con_lin_time = None
        self.con_dyn = None
        self.con_time = None

    def set_eco(self, e):
        self.eco = e

    def set_eco_param(self, d, z, J, V, theta, eps, directed=False, ntype=None, A=None):
        self.eco = Economy(self.n, d, z, J, V, theta)
        if A is None:
            if ntype is None:
                raise InputError('Specify parameters of the network if adjacency matrix not provided')
            else:
                self.eco.gen_M(eps=eps, directed=directed, ntype=ntype)
        else:
            self.eco.set_M_eps(eps=eps, A=A)

        self.eco.compute_p()
        self.eco.compute_lda()

    def discrete_step(self, prev, cur, alpha, beta, q):
        mu = 1. / (self.n * np.nanmean(self.eco.theta))
        if len(prev) % 2 != 0 or len(cur) % 2 != 0:
            raise ValueError('Arguments lengths must be divisible by 2')
        pp, lp = np.split(prev, 2)
        pc, lc = np.split(cur, 2)

        l_new = lc + beta * mu * self.eco.theta / (self.eco.z * pc) - beta * lc * self.eco.V / (
                self.eco.z * pc) - beta * lc * np.array(np.dot(
            np.multiply(self.eco.J, self.eco.A), pp.T)) / (self.eco.z * pc) + beta * np.array(np.dot(np.multiply(self.eco.J, self.eco.A).T, np.multiply(lc, np.power(lc / lp, q)).T))[0] / self.eco.z

        p_new = pc - alpha * pc * lp / lc + alpha * (pc / (self.eco.z * lc)) * np.array(np.dot(
            np.multiply(self.eco.J, self.eco.A).T,
            np.multiply(lc, np.power(lc / lp, q)).T))[0] + alpha * mu * self.eco.theta / (self.eco.z * lc)
        if self.eco.directed:
            return np.concatenate((p_new, l_new[0]))
        else:
            return np.concatenate((p_new, l_new))

    def discrete_dyn(self, end, x0, x1, alpha, beta, q):
        a, b = x0, x1
        track = [x0, x1]
        k = 2
        while k < end:
            k += 1
            xn = self.discrete_step(a, b, alpha, beta, q)
            a = b
            b = xn
            track.append(xn)
        self.dis_dyn = np.array(track)

    def linear_discrete_step(self, prev, cur):
        xn = np.array(np.dot(self.eco.Sd, np.concatenate((cur, prev))))[0]
        #print(xn)
        return np.split(xn, 2)[0]

    def linear_discrete_dyn(self, end, x0, x1):
        a, b = x0, x1
        track = [x0, x1]
        k = 0
        while k < end:
            k += 1
            xn = self.linear_discrete_step(a, b)
            a = b
            b = xn
            track.append(xn)
        self.dis_lin_dyn = np.array(track)

    def linear_cont_dyn(self, end, pert0):
        run = sci.RK45(lambda t, x: np.dot(self.eco.Sc, x.T), 0, pert0, end, max_step=1)
        time = [0.]
        pert = [pert0]
        while run.status == 'running':
            run.step()
            time.append(run.t)
            pert.append(run.y)
        if run.status == 'failed':
            print('Failed')
        else:
            self.con_lin_dyn = np.array(pert)
            self.con_lin_time = np.array(time)

    def cont_step(self, t, x, alpha, beta):
        mu = 1. / (self.n * np.nanmean(self.eco.theta))
        p, l = np.split(x, 2)

        # Auxiliary variables
        psi = np.divide(self.eco.theta, p)
        dummy1 = np.divide(p, self.eco.z * l)
        dummy2 = np.divide(l, self.eco.z * p)

        dp_dt = -alpha * np.dot(np.einsum('i,ji->ij', dummy1, self.eco.M), l.T) + alpha * mu * psi * dummy1

        dl_dt = beta * (- np.dot(np.einsum('i,ij->ij', dummy2, self.eco.J * self.eco.A), p.T) /
                        + np.dot(np.einsum('i,ji->ij', 1./self.eco.z, self.eco.J * self.eco.A), l.T)) /\
                        - beta * dummy2 * self.eco.V + beta * mu * psi / self.eco.z
        return np.concatenate((dp_dt, dl_dt))

    def cont_dyn(self, end, x0, alpha, beta):
        run = sci.RK45(lambda t, x: self.cont_step(t, x, alpha, beta), 0, x0, end)
        time = [0.]
        x = [x0]
        while run.status == 'running':
            run.step()
            time.append(run.t)
            x.append(run.y)
        if run.status == 'failed':
            print('Failed')
        else:
            self.con_dyn = np.array(x)
            self.con_time = np.array(time)

    def profit_dyn(self, cont=False):
        mu = 1. / (self.n * np.nanmean(self.eco.theta))
        # profit = None
        if cont:
            tr = self.con_dyn
        else:
            tr = self.dis_dyn
        profit = np.array([tr[t, :self.n] * np.dot(np.multiply(self.eco.J, self.eco.A).T, np.roll(tr, -1)[t, self.n:]) + mu * self.eco.theta - tr[t, self.n:] * np.dot(
            np.multiply(self.eco.J, self.eco.A), tr[t, :self.n]) - self.eco.V for t in range(len(tr))])
        return profit

    def surplus_dyn(self, cont=False):
        mu = 1. / (self.n * np.nanmean(self.eco.theta))
        if cont:
            tr = self.con_dyn
        else:
            tr = self.dis_dyn
        roll1 = np.roll(tr, 1)
        roll_1 = np.roll(tr, -1)
        surplus = np.array([self.eco.z * roll1[t, self.n:] - np.dot(np.multiply(self.eco.J, self.eco.A).T, roll_1[t, self.n:]) - mu * self.eco.theta / tr[t, :self.n] for t in range(len(tr))])
        return surplus

class Economy_ces(object):

    def __init__(self, n, z, j, a, q, b, theta ):
        self.n = n #number of firms
        self.z = z #productivity of firms
        self.j = j #input-output network
        self.a = a #substitution network
        self.q = q #CES interpolator
        self.zeta = 1/(self.q+1) #recurrent parameter
        self.b = b #return to scale
        self.theta = theta #preferences vector
        self.v = None
        self.mu = 1./np.sum(self.theta)
        self.m = None
        self.p_eq = None
        self.g_eq = None
        self.eps = None
        self.i = None

    def set_v(self, a0, j0):
        self.v = np.power(self.b * a0 * self.z, self.q * self.zeta) * np.power(j0, self.q * self.zeta)

    def set_eps(self):
        if self.q != 0:
            self.i = np.power(self.b * np.diag(self.z) * self.a, self.q * self.zeta) \
                * np.power(self.j, self.zeta)
        else:
            self.i = self.j
        self.m = np.diag(self.z) - self.i
        self.eps = np.max(self.z) - np.max(np.linalg.eigvals(self.i))

    def m_eigvals(self):
        return np.linalg.eigvals(self.m)

    def m_eigvecs(self):
        return np.linalg.eig(self.m)[1]

    def m_spectrum(self):
        return np.linalg.eig(self.m)

    def compute_eq(self):
        # Inital values for iterative algo (b=1)
        if self.eps > 0:
            init_guess_u = np.array(scla.lstsq(self.m, self.v)[0].reshape(self.n))
            init_guess_g = np.array(scla.lstsq(self.m.T * np.power(np.diag(init_guess_u), \
                        self.q), self.mu * self.theta / init_guess_u)[0].reshape(self.n))
        else:
            init_guess_u = np.ones(self.n)
            init_guess_g = np.ones(self.n)

        x_eq = scop.fsolve(eq_root_function, np.concatenate((init_guess_u, init_guess_g)) ,
                         args=(self.z, self.m, self.b, self.q, self.v, self.mu, self.theta))
        print(x_eq)
        p_eq, self.g_eq = np.split(x_eq, 2)
        self.p_eq = np.power(p_eq, 1./self.zeta)

class Dynamics_ces(object):

    def __init__(self, n, alpha, beta):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.eco = None
        self.dis_dyn = None
        #self.con_dyn = None
        #self.dis_lin_dyn = None
        #self.con_lin_dyn = None
        #self.con_lin_time = None
        #self.con_dyn = None
        #self.con_time = None

    def set_eco(self, e):
        self.eco = e

    def discrete_prof(self, p_cur, p_prev, g_cur, g_prev):
        gain = self.eco.mu * self.eco.theta + np.multiply(np.power(p_cur, self.eco.zeta), np.array(np.dot(self.eco.i.T, \
                                np.power(p_cur, self.eco.q * self.eco.zeta) * \
                                np.power(g_cur, self.eco.zeta * (self.eco.b * self.eco.q + 1)/self.eco.b)).reshape(self.n)))
        loss = self.eco.v * np.power(p_prev, self.eco.q * self.eco.zeta) * \
               np.power(g_prev, self.eco.zeta * (self.eco.b * self.eco.q + 1)/self.eco.b) + \
               np.power(p_prev, self.eco.q * self.eco.zeta) * np.power(g_prev, self.eco.zeta * (
                    self.eco.b * self.eco.q + 1) / self.eco.b) * np.array(np.dot(self.eco.i, np.power(p_prev, self.eco.zeta)).reshape(self.n))
        return np.array((gain - loss).reshape(self.n))

    def discrete_balance(self, p_cur, p_prev, g_cur, g_prev):
        return np.array((self.eco.z * g_prev -  (self.eco.mu * self.eco.theta / p_cur + \
                                np.multiply(np.power(p_cur, self.eco.zeta), np.array(np.dot(self.eco.i.T, \
                                np.power(p_cur, self.eco.q * self.eco.zeta) * \
                                np.power(g_cur, self.eco.zeta * (self.eco.b * self.eco.q + 1)/self.eco.b)).reshape(self.n))) / p_cur).reshape(self.n)))

    def discrete_update(self, x_prev, x_cur):
        p_cur, g_cur = np.split(x_cur, 2)
        p_prev, g_prev = np.split(x_prev, 2)
        balance = self.discrete_balance(p_cur, p_prev, g_cur, g_prev)
        profit = self.discrete_prof(p_cur, p_prev, g_cur, g_prev)
        p_new = p_cur * (1 - self.alpha * balance / (self.eco.z * g_cur))
        g_new = g_cur * (1 + self.beta * profit / (self.eco.z * p_cur * g_cur))
        return np.concatenate((p_new, g_new))

    def discrete_dyn(self, T, x0, x1):
        a, b = x0, x1
        track = [x0, x1]
        k = 2
        while k < T:
            k += 1
            xn = self.discrete_update(a, b)
            a = b
            b = xn
            track.append(xn)
        print(k)
        self.dis_dyn = np.array(track)

