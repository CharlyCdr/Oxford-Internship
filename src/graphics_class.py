import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import seaborn as sns


# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.ticklabel_format(axis='y', scilimits=(2,2))
# plt.rcParams['axes.xmargin'] = 0


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
                #self.axs_firms[0].set_ylim(-2 * 10e-5, 2 * 10e-5)
                #self.axs_firms[1].set_ylim(-2 * 10e-5, 2 * 10e-5)

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

            self.fig_firms.show()

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

        #for i, col in enumerate(self.cons_color(self.dyn.eco.house.theta)):
        self.axs_house[0, 0].plot(self.dyn.Q_real[1:-1, 0, 1:])
        self.axs_house[0, 1].plot(self.dyn.utility[1:-1])
        self.axs_house[1, 0].plot(self.dyn.budget[1:-1])
        self.axs_house[1, 1].plot(self.dyn.wages[1:-1])

        self.fig_firms.show()