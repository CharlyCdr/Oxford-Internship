import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class PlotlyDynamics(object):

    def __init__(self, dyn):
        self.dyn = dyn

        self.rc = {"text.usetex": True,
                   "font.family": 'serif',
                   "font.size": 12,
                   "axes.grid": True,
                   "grid.alpha": 0.5,
                   "axes.xmargin": 0.025,
                   "axes.ymargin": 0.05,
                   }
        self.norm_cbar = mpl.colors.Normalize(vmin=0, vmax=1)

        self.prices_label = r'$\tilde{p}_{i}(t)$'
        self.prods_label = r'$\gamma_{i}(t)$'
        self.stocks_label = r'$s_{i}(t)$'

        self.budget_label = r'$\tilde{B}(t)$'
        self.cons_label = r'$C_{i}(t)$'
        self.utility_label = r'$\mathcal{U}(t)$'
        self.wage_label = r'$p_{0}(t)$'

        self.color_firms = np.array(sns.color_palette("deep", self.dyn.n))
        self.stocks_color = ListedColormap(sns.color_palette("PuBuGn_d", n_colors=100).as_hex())
        self.cons_color = ListedColormap(sns.color_palette("Greens_d", n_colors=100).as_hex())

        self.fig_firms_funda = None
        self.fig_house = None
        self.fig_network = None
        self.fig_firms_observ = None

    def plotNetwork(self):
        edge_x = []
        edge_y = []
        G = nx.from_numpy_matrix(self.dyn.eco.j)
        pos = nx.spectral_layout(G)
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_z = []
        node_text = []
        for node, adjacencies in enumerate(G.adjacency()):
            node_z.append(self.dyn.eco.firms.z[node])
            node_text.append('Productivity factor: ' + str(self.dyn.eco.firms.z[node]))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                # colorscale options
                # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))

        node_trace.marker.color = node_z
        node_trace.text = node_text
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            annotations=[dict(
                                showarrow=True,
                                xref="paper", yref="paper",
                                x=edge_x, y=edge_y)],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        self.fig_network = fig

    def stockTrace(self, stocks, firms):
        cols = list(zip(self.stocks_color(self.dyn.eco.firms.sigma[firms]),
                        self.color_firms[firms]))
        trace = go.Figure()
        for k in firms:
            trace.add_trace(go.Scatter(x=np.arange(self.dyn.t_max),
                            y=stocks[1:, k],
                           mode='lines'))
        return trace

    def plotFirms(self, from_eq=None, k=None):
        if k:
            firms = np.random.choice(self.dyn.n, k, replace=False)
        else:
            firms = np.arange(self.dyn.n)

        stock_trace = self.stockTrace(self.dyn.stocks, firms)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02
                            )
        fig.update_xaxes(title_text=r'$t$', row=3, col=1)
        fig.update_yaxes(title_text=r'$p_{i}(t)$', row=1, col=1)
        fig.update_yaxes(title_text=r'$\gamma_{i}(t)$', row=2, col=1)
        fig.update_yaxes(title_text=r'$s_{i}(t)$', row=3, col=1)
        for l in firms:
            if from_eq:
                fig.add_trace(go.Scatter(x=np.arange(self.dyn.t_max),
                                         y=self.dyn.prices[1:, l]-self.dyn.eco.p_eq[l],
                                         mode='lines'),
                              row=1, col=1)
                fig.add_trace(go.Scatter(x=np.arange(self.dyn.t_max),
                                         y=self.dyn.prods[1:, l]-self.dyn.eco.g_eq[l],
                                         mode='lines'),
                              row=2, col=1)
                fig.add_trace(go.Scatter(x=np.arange(self.dyn.t_max),
                                         y=self.dyn.stocks[1:, l],
                                         mode='lines'),
                              row=3, col=1)
            else:
                fig.add_trace(go.Scatter(x=np.arange(self.dyn.t_max),
                                         y=self.dyn.prices[1:, l],
                                         mode='lines'),
                              row=1, col=1)
                fig.add_trace(go.Scatter(x=np.arange(self.dyn.t_max),
                                         y=self.dyn.prods[1:, l],
                                         mode='lines'),
                              row=2, col=1)
                fig.add_trace(go.Scatter(x=np.arange(self.dyn.t_max),
                                         y=self.dyn.stocks[1:, l],
                                         mode='lines'),
                              row=3, col=1)
        fig.update_layout(showlegend=False)
        self.fig_firms_funda = fig


class PlotDynamics(object):

    def __init__(self, dyn):
        self.dyn = dyn

        self.rc = {"text.usetex": True,
                   "font.family": 'serif',
                   "font.size": 12,
                   "axes.grid": True,
                   "grid.alpha": 0.5,
                   "axes.xmargin": 0.025,
                   "axes.ymargin": 0.05,
                   }
        self.norm_cbar = mpl.colors.Normalize(vmin=0, vmax=1)

        self.prices_label = r'$\tilde{p}_{i}(t)$'
        self.prods_label = r'$\gamma_{i}(t)$'
        self.stocks_label = r'$s_{i}(t)$'

        self.budget_label = r'$\tilde{B}(t)$'
        self.cons_label = r'$C_{i}(t)$'
        self.utility_label = r'$\mathcal{U}(t)$'
        self.wage_label = r'$p_{0}(t)$'

        self.color_firms = np.array(sns.color_palette("deep", 100))
        self.stocks_color = ListedColormap(sns.color_palette("PuBuGn_d", n_colors=100).as_hex())
        self.cons_color = ListedColormap(sns.color_palette("Greens_d", n_colors=100).as_hex())

        self.fig_firms, self.axs_firms = None, None
        self.fig_house, self.axs_house = None, None

    def layout_firms(self):
        with plt.rc_context(rc=self.rc):
            self.fig_firms, self.axs_firms = plt.subplots(figsize=(8, 7),
                                                          nrows=3,
                                                          ncols=1,
                                                          tight_layout=True
                                                          )
            self.axs_firms[0].ticklabel_format(axis='y', scilimits=(-1, 1))
            self.axs_firms[1].ticklabel_format(axis='y', scilimits=(-1, 1))
            self.axs_firms[2].ticklabel_format(axis='y', scilimits=(-1, 1))

            divider0 = make_axes_locatable(self.axs_firms[0])
            cax0 = divider0.append_axes('right', size='5%', pad=0.05)
            cax0.axis('off')
            divider1 = make_axes_locatable(self.axs_firms[1])
            cax1 = divider1.append_axes('right', size='5%', pad=0.05)
            cax1.axis('off')
            divider2 = make_axes_locatable(self.axs_firms[2])
            cax2 = divider2.append_axes('right', size='5%', pad=0.05)
            cax2.set_yticks([0, 1])
            cax2.yaxis.set_label_position('right')

            self.axs_firms[0].set_ylabel(self.prices_label)
            self.axs_firms[1].set_ylabel(self.prods_label)
            self.axs_firms[2].set_ylabel(self.stocks_label)
            self.axs_firms[-1].set_xlabel(r'$t$', rotation=0)
            cb = mpl.colorbar.ColorbarBase(cax2,
                                           cmap=self.stocks_color,
                                           norm=self.norm_cbar,
                                           ticks=[0, 1],
                                           orientation='vertical'
                                           )
            cb.set_label(r'$\sigma_{i}$', rotation=0)

    def plotFirms(self, from_eq=False, k=None):
        self.layout_firms()

        if k:
            firms = np.random.choice(self.dyn.n, k, replace=False)
            self.axs_firms[0].set_title(str(k) + r' randomly chosen firms')
        else:
            firms = np.arange(self.dyn.n)
            self.axs_firms[0].set_title(r'Every firms')

        cols = list(zip(self.stocks_color(self.dyn.eco.firms.sigma[firms]),
                        self.color_firms[firms]))
        if from_eq:
            # self.axs_firms[0].set_ylim(-2 * 10e-5, 2 * 10e-5)
            # self.axs_firms[1].set_ylim(-2 * 10e-5, 2 * 10e-5)

            p_eq = self.dyn.eco.p_eq
            g_eq = self.dyn.eco.g_eq

            for i, firm_num in enumerate(firms):
                self.axs_firms[0].plot(self.dyn.prices[1:, firm_num] - p_eq[firm_num], color=cols[i][1])
                self.axs_firms[1].plot(self.dyn.prods[1:, firm_num] - g_eq[firm_num], color=cols[i][1])
                self.axs_firms[2].plot(self.dyn.stocks[1:, firm_num], color=cols[i][0])
        else:
            for i, firm_num in enumerate(firms):
                self.axs_firms[0].plot(self.dyn.prices[1:, firm_num], color=cols[i][1])
                self.axs_firms[1].plot(self.dyn.prods[1:, firm_num], color=cols[i][1])
                self.axs_firms[2].plot(self.dyn.stocks[1:, firm_num], color=cols[i][0])

        # self.fig_firms.show()

    def layout_household(self):
        with plt.rc_context(rc=self.rc):
            self.fig_house, self.axs_house = plt.subplots(figsize=(8, 8),
                                                          nrows=2,
                                                          ncols=2,
                                                          tight_layout=True
                                                          )

            self.axs_house[0, 0].ticklabel_format(axis='y', scilimits=(-1, 1))
            self.axs_house[1, 0].ticklabel_format(axis='y', scilimits=(-1, 1))
            self.axs_house[0, 1].ticklabel_format(axis='y', scilimits=(-1, 1))
            self.axs_house[1, 1].ticklabel_format(axis='y', scilimits=(-1, 1))

            self.axs_house[0, 0].set_ylabel(self.cons_label)
            self.axs_house[0, 1].set_ylabel(self.utility_label)
            self.axs_house[1, 0].set_ylabel(self.budget_label)
            self.axs_house[1, 1].set_ylabel(self.wage_label)
            self.axs_house[1, 0].set_xlabel(r'$t$', rotation=0)
            self.axs_house[1, 1].set_xlabel(r'$t$', rotation=0)

            divider0 = make_axes_locatable(self.axs_house[0, 0])
            cax0 = divider0.append_axes('right', size='5%', pad=0.05)
            cax0.set_yticks([0, 1])
            cax0.yaxis.set_label_position('right')
            divider1 = make_axes_locatable(self.axs_house[0, 1])
            cax1 = divider1.append_axes('right', size='5%', pad=0.05)
            cax1.axis('off')
            divider2 = make_axes_locatable(self.axs_house[1, 0])
            cax2 = divider2.append_axes('right', size='5%', pad=0.05)
            cax2.axis('off')
            divider3 = make_axes_locatable(self.axs_house[1, 1])
            cax3 = divider3.append_axes('right', size='5%', pad=0.05)
            cax3.axis('off')

            cb = mpl.colorbar.ColorbarBase(cax0,
                                           cmap=self.cons_color,
                                           norm=self.norm_cbar,
                                           ticks=[0, 1],
                                           orientation='vertical'
                                           )
            cb.set_label(r'$\theta_{i}$', rotation=0)

    def plotHousehold(self):
        self.layout_household()

        # for i, col in enumerate(self.cons_color(self.dyn.eco.house.theta)):
        self.axs_house[0, 0].plot(self.dyn.Q_real[1:-1, 0, 1:])
        self.axs_house[0, 1].plot(self.dyn.utility[1:-1])
        self.axs_house[1, 0].plot(self.dyn.budget[1:-1])
        self.axs_house[1, 1].plot(self.dyn.wages[1:-1])

        # self.fig_firms.show()
