"""
The ``network`` module
======================

This module deals with various network generation useful for the model.
"""
import networkx as nx
import numpy as np


def undir_rrg(d, n):
    """
    Generates an undirected d-regular network on n nodes.
    :param d: node connectivity,
    :param n: number of nodes.
    :return: Adjacency matrix of the network.
    """
    return np.array(nx.convert_matrix.to_numpy_array(nx.random_regular_graph(d, n)))


def dir_rrg(d, n):
    """
    Generates a directed d-regular network on n nodes with in and out connectivity d. Note: bad implementation but
    still really efficient because of very few cases for which the generated network does not satisfy the while
    loop condition.
    :param d: node connectivity,
    :param n: number of nodes.
    :return: Adjacency matrix of the network.
    """
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
    """
    Generates a directed regular network with in and out average connectivity d.
    :param d: average node connectivity,
    :param n: number of nodes.
    :return: Adjacency matrix of the network.
    """
    A1 = nx.convert_matrix.to_numpy_matrix(nx.random_regular_graph(d, n))
    A2 = nx.convert_matrix.to_numpy_matrix(nx.random_regular_graph(d, n))
    return np.triu(A1) + np.tril(A2)


def er(n, p, directed=False):
    """
    Generates an undirected (or directed) Erdös-Renyi network with link probability p.
    :param p: probability for link presence,
    :param n: number of nodes,
    :param directed: whether or not the network is directed, default False.
    :return: Adjacency matrix of the network.
    """
    return np.array(nx.convert_matrix.to_numpy_array(nx.binomial_graph(n, p, directed=directed)))


def create_net(net_str, directed, n, d):
    """
    Generates the prescribed network.
    :param net_str: type of network - 'regular' for regular, 'm-regular' for multi-regular, 'er' for Erdös-Renyi,
    :param directed: whether or not the network is directed,
    :param n: number of nodes,
    :param d: average connectivity.
    :return: Adjacency matrix of the network.
    """
    if directed:
        if net_str == 'regular':
            return dir_rrg(d, n)
        elif net_str == 'm_regular':
            return mdir_rrg(d, n)
        elif net_str == 'er':
            return er(n, d / n, directed=directed)
    else:
        if net_str == 'regular':
            return undir_rrg(d, n)
        elif net_str == 'er':
            return er(n, d / n)
        else:
            raise Exception("Not coded yet")


# Graphical representation of networks from stack overflow
# https://stackoverflow.com/questions/43541376/how-to-draw-communities-with-networkx

def community_layout(g, partition):
    """
    Computes the layout for a modular graph.
    :param g: networkx.Graph or networkx.DiGraph instance graph to plot,
    :param partition: dict mapping int node -> int community graph partitions.
    :return: dict mapping int node -> (float x, float y) node positions.
    """
    pos_communities = _position_communities(g, partition, scale=3.)
    pos_nodes = _position_nodes(g, partition, scale=1.)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]
    return pos


def _position_communities(g, partition, **kwargs):
    """
    Creates a weighted graph, in which each node corresponds to a community,
    and each edge weight to the number of edges between communities
    :param g: networkx.Graph or networkx.DiGraph instance graph to plot,
    :param partition: dict mapping int node -> int community graph partitions,
    :return: dict mapping int node -> (float x, float y) community positions.
    """
    between_community_edges = _find_between_community_edges(g, partition)
    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]
    return pos


def _find_between_community_edges(g, partition):
    """
    Determines edges between communities.
    :param g: networkx.Graph or networkx.DiGraph instance graph to plot,
    :param partition: dict mapping int node -> int community graph partitions,
    :return: dict mapping tuple (community, community) -> (float x, float y) edges.
    """
    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges


def _position_nodes(g, partition, **kwargs):
    """"
    Positions nodes within communities.
    :param g: networkx.Graph or networkx.DiGraph instance graph to plot,
    :param partition: dict mapping int node -> int community graph partitions,
    :return: dict mapping int node -> (float x, float y) node positions.
    """
    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos
