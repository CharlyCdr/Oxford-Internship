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


class PlotEconomy(object):

    @staticmethod
    def utility_df(df_cons, df_work, theta, l, gamma, phi):
        return (theta * np.log(df_cons)).sum(axis=1) - gamma * np.power(df_work / l, 1 + phi) / (1 + phi)

    def __init__(self, dfeco, dfdyn):
        '''
        try:
            assert self.dfdyn.index.levels[1].to_list() == ['Budget/Stocks', 'Cons', 'Demand', 'Labour/Productions','Mu/Targets','Wage/Prices']
            assert self.dfeco.index.levels[0].to_list() == ['Firms', 'Household']
            assert self.dfeco.index.levels[1].to_list() == ['alpha', 'alpha_p','b','beta','beta_p','gamma','l','phi','q','sigma','theta','w','z']
            self.dfeco = dfeco
            self.dfdyn = dfdyn
        except Exception as e:
            print("Wrong second indexing format for either dfeco or dfdyn")
        '''

        self.dfeco = dfeco
        self.dfdyn = dfdyn
        self.index_wage = 'Wage/Prices'
        self.index_prices = 'Wage/Prices'
        self.index_labour = 'Labour/Productions'
        self.index_prods = 'Labour/Prods'
        self.index_budget = 'Budget/Stocks'
        self.index_stocks = 'Budget/Stocks'
        self.index_mu = 'Mu/Targets'
        self.index_targets = 'Mu/Targets'
        self.index_cons = 'Cons'
        self.index_demand = 'Demand'

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

        self.budget_label = r'$\tilde{B}(t)'
        self.cons_label = r'$C_{i}(t)$'
        self.utility_label = r'$\mathcal{U}(t)$'
        self.wage_label = r'$p_{0}(t)'

        self.stocks_color = ListedColormap(sns.color_palette("PuBuGn_d").as_hex())

        self.fig_firms, self.axs_firms = None, None
        self.fig_house, self.axs_house = None, None

    def plotFirms_from_dataframes(self, from_eq=False, k=None):
        with plt.rc_context(rc=self.rc):
            self.fig_firms, self.axs_firms = plt.subplots(figsize=(7, 6),
                                                          nrows=3,
                                                          ncols=1,
                                                          tight_layout=True
                                                          )
            if k:
                firms = np.random.choice(len(self.dfdyn.columns) - 1, k, replace=False)
                self.axs_firms[0].set_title(str(k) + r' randomly chosen firms')
            else:
                firms = np.arange(1, len(self.dfdyn.columns))
                self.axs_firms[0].set_title(r'Every firms')
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

            if from_eq:
                self.axs_firms[0].set_ylim(-2 * 10e-5, 2 * 10e-5)
                self.axs_firms[1].set_ylim(-2 * 10e-5, 2 * 10e-5)

                p_eq = self.dfeco.loc[pd.IndexSlice['Firms', 'p_eq']][firms - 1]
                g_eq = self.dfeco.loc[pd.IndexSlice['Firms', 'g_eq']][firms - 1]
                self.axs_firms[0].plot((self.dfdyn.loc[pd.IndexSlice[:, 'Wage/Prices'], firms] - p_eq).values)
                self.axs_firms[1].plot((self.dfdyn.loc[pd.IndexSlice[:, 'Labour/Productions'], firms] - g_eq).values)
                for i, col in enumerate(self.stocks_color(self.dfeco.loc[pd.IndexSlice['Firms', 'sigma']][firms - 1])):
                    self.axs_firms[2].plot(self.dfdyn.loc[pd.IndexSlice[:, 'Budget/Stocks'], i].values, color=col)
            else:
                self.axs_firms[0].plot(self.dfdyn.loc[pd.IndexSlice[:, 'Wage/Prices'], firms].values)
                self.axs_firms[1].plot(self.dfdyn.loc[pd.IndexSlice[:, 'Labour/Productions'], firms].values)
                for i, col in enumerate(self.stocks_color(self.dfeco.loc[pd.IndexSlice['Firms', 'sigma']][firms - 1])):
                    self.axs_firms[2].plot(self.dfdyn.loc[pd.IndexSlice[:, 'Budget/Stocks'], i].values, color=col)

            cb = mpl.colorbar.ColorbarBase(cax2,
                                           cmap=self.stocks_color,
                                           norm=self.norm_cbar,
                                           ticks=[0, 1],
                                           orientation='vertical'
                                           )
            cb.set_label(r'$\sigma_{i}$', rotation=0)
            self.fig_firms.show()

    def plotHousehold_from_dataframes(self):

        with plt.rc_context(rc=self.rc):
            self.fig_house, self.axs_house = plt.subplots(figsize=(7, 7),
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

            dfwork = self.dfdyn.loc[pd.IndexSlice[:, 'Demand'], 0] * np.clip(
                self.dfdyn.loc[pd.IndexSlice[:, 'Labour/Productions'], 0].values /
                self.dfdyn.loc[pd.IndexSlice[:, 'Demand'], 0].values, None, 1)

            dfutil = self.utility_df(self.dfdyn.loc[pd.IndexSlice[:, 'Cons'], 1:],
                                     dfwork,
                                     self.dfeco.loc[pd.IndexSlice['Household', 'theta']].values,
                                     self.dfeco.loc[pd.IndexSlice['Household', 'l'], 1],
                                     self.dfeco.loc[pd.IndexSlice['Household', 'gamma'], 1],
                                     self.dfeco.loc[pd.IndexSlice['Household', 'phi'], 1]
                                     )

            dfbudget = self.dfdyn.loc[pd.IndexSlice[:, 'Budget/Stocks'], 0].shift(-1) + dfwork

            self.axs_house[0, 0].plot(self.dfdyn.loc[pd.IndexSlice[:, 'Cons'], 1:].values)
            self.axs_house[0, 1].plot(dfbudget)
            self.axs_house[1, 0].plot(dfutil)
            self.axs_house[1, 1].plot(self.dfdyn.loc[pd.IndexSlice[:, 'Wage/Prices'], 0].values)

            self.fig_firms.show()
    # def plotFirms_from_class(self, dyn):
    #


def plot_firms_fundamentals(dyn, k=None):
    """

    :param prices:
    :param prods:
    :param stocks:
    :param k:
    :return:
    """

    f, axs = plt.subplots(figsize=(7, 6), nrows=3, ncols=1, tight_layout=True)
    # plt.grid(axis='both')

    # Layout
    axs[0].grid(axis='both', alpha=.5)
    axs[1].grid(axis='both', alpha=.5)
    axs[2].grid(axis='both', alpha=.5)

    # Title and labels
    axs[0].set_title(r'Firms fundamentals')
    axs[0].set_ylabel(r'$\tilde{p}_{i}(t)$', rotation=0)
    axs[1].set_ylabel(r'$\gamma_{i}(t)$', rotation=0)
    axs[2].set_ylabel(r'$s_{i}(t)$', rotation=0)

    n = dyn.eco.n
    if k:
        ind = np.random.choice(n, k, replace=False)
    else:
        ind = range(n)

    # color_stocks
    for l in ind:
        axs[0].plot(dyn.prices[1:-1, l])
        axs[1].plot(dyn.prods[:-1, l])
        axs[2].plot(dyn.stocks[:-1, l], color=(dyn.eco.firms.sigma[l], 0, 1 - dyn.eco.firms.sigma[l]))

    plt.show()
