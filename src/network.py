import networkx as nx
import numpy as np
from exception import *


def undir_rrg(d, n):
    return np.array(nx.convert_matrix.to_numpy_array(nx.random_regular_graph(d, n)))


def dir_rrg(d, n):
    A = np.zeros((n, n))
    while not ((np.sum(A, axis=0) == d).all() and (np.sum(A, axis=1) == d).all()):
        A = np.zeros((n, n))
        ind = np.random.choice(np.arange(1, n), d, replace=False)
        A[0, ind] = 1
        for k in range(1, n):
            sums = np.sum(A, axis=0)
            m = np.where(np.min(sums) == sums)[0]
            m = m[m != k]
            if len(m) < d:
                r = len(m)
                it = r
                it_aux = 1
                while it < d:
                    maux = np.where(np.min(sums) + it_aux == sums)[0]
                    maux = maux[maux != k]
                    if len(maux) < d - it:
                        m = np.append(m, maux)
                    else:
                        aux = np.random.choice(maux, d - it, replace=False)
                        m = np.append(m, aux)
                    it += min(len(maux), d - it)
                    it_aux += 1
                ind = m
            else:
                ind = np.random.choice(m, d, replace=False)
            A[k, ind] = 1
    return A


def mdir_rrg(d, n):
    A1 = nx.convert_matrix.to_numpy_matrix(nx.random_regular_graph(d, n))
    A2 = nx.convert_matrix.to_numpy_matrix(nx.random_regular_graph(d, n))
    return np.triu(A1) + np.tril(A2)


def er(n, p, directed=False):
    return np.array(nx.convert_matrix.to_numpy_array(nx.binomial_graph(n, p, directed=directed)))


def create_net(net_str, directed, n, d):
    if directed:
        if net_str == 'regular':
            return dir_rrg(d, n)
        elif net_str == 'm_regular':
            return mdir_rrg(d, n)
        elif net_str == 'er':
            raise InputError("Not coded yet for Erdos-Renyi directed network")
    else:
        if net_str == 'regular':
            return undir_rrg(d, n)
        elif net_str == 'er':
            return er(n, d/n)
        else:
            raise InputError("Not coded yet")
