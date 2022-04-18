import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import cm

from utils import lists_union


class Plotter(object):
    def __init__(self, data_path, plots_path):
        self.data_path = data_path
        self.plots_path = plots_path
    
    @staticmethod
    def _crop_data_table(data_table, crop=None):
        if crop is None:
            return data_table
        
        xlabel = data_table.columns[0]
        
        min_value, max_value = crop
        
        if min_value is None:
            min_value = min(data_table[xlabel])
        
        if max_value is None:
            max_value = max(data_table[xlabel])
        
        return data_table[(data_table[xlabel] >= min_value) & (data_table[xlabel] <= max_value)]

    @staticmethod
    def set_ax_params(ax, title, xlabel, ylabel, xticks, fontsize=20):
        ax.set_title(title, fontsize=fontsize)
        
        ax.set_xticks(xticks)

        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)

        ax.tick_params(axis='x', which='major', labelsize=fontsize - 4, rotation=45)
        ax.tick_params(axis='y', which='major', labelsize=fontsize - 4)

        ax.legend(loc='best', prop={'size': fontsize - 6})

        ax.grid(True, alpha=0.5)

    @staticmethod
    def plot_indicator(ax, data_table, label, **kwargs):
        xlabel = data_table.columns[0]
        ylabel = data_table.columns[1]

        ax.plot(data_table[xlabel], data_table[ylabel], label=label, **kwargs)

    def _get_regions(self, indicator, regions=None):
        if regions is None:
            regions = os.listdir(self.data_path)

        correct_regions = []
        for region in regions:
            if os.path.exists(os.path.join(self.data_path, region, indicator + '.csv')):
                correct_regions.append(region)

        return correct_regions

    @staticmethod
    def save_plot(path, plot_name):
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, plot_name) + '.pdf', bbox_inches='tight')
    
    def plot_indicators(self, region, indicators, crop=None, title=None, colors=None, save=False):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
        
        if colors is None:
            colors = [cm.gist_rainbow(value) for value in np.linspace(start=0.0, stop=1.0, num=len(indicators))]
        colors = colors[:len(indicators)]
        
        xlabels = []
        ylabels = []
        xticks = []
        for indicator, color in zip(indicators, colors):
            data_table = self._crop_data_table(pd.read_csv(os.path.join(self.data_path, region, indicator + '.csv')).dropna(),
                                               crop=crop)
            
            self.plot_indicator(ax, data_table, label=indicator, **{'color': color, 'alpha': 1.0,
                                                                    'linewidth': 2, 'linestyle': '-',
                                                                    'marker': 'o',
                                                                    'markerfacecolor': color, 'markeredgecolor': 'black'})
            
            xlabels.append(data_table.columns[0])
            ylabels.append(data_table.columns[1])
            xticks.append(list(data_table[xlabels[-1]]))
        
        if title is None:
            title = region
        
        self.set_ax_params(ax, title=title,
                           xlabel=lists_union(xlabels),
                           ylabel=lists_union(ylabels),
                           xticks=lists_union(xticks),
                           fontsize=20)
        
        if save:
            self.save_plot(path=self.plots_path, plot_name=f'{region}_' + ('_').join(indicators))
    
    def plot(self, indicator, regions=None, crop=None, title=None, colors=None, save=False):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
        
        regions = self._get_regions(indicator=indicator, regions=regions)
        
        if colors is None:
            colors = [cm.gist_rainbow(value) for value in np.linspace(start=0.0, stop=1.0, num=len(regions))]
        colors = colors[:len(regions)]
        
        xlabels = []
        ylabels = []
        xticks = []
        for region, color in zip(regions, colors):
            data_table = self._crop_data_table(pd.read_csv(os.path.join(self.data_path, region, indicator + '.csv')).dropna(),
                                               crop=crop)
            
            self.plot_indicator(ax, data_table, label=region, **{'color': color, 'alpha': 1.0,
                                                                 'linewidth': 2, 'linestyle': '-',
                                                                 'marker': 'o',
                                                                 'markerfacecolor': color, 'markeredgecolor': 'black'})
            
            xlabels.append(data_table.columns[0])
            ylabels.append(data_table.columns[1])
            xticks.append(list(data_table[xlabels[-1]]))
        
        if title is None:
            title = indicator
        
        self.set_ax_params(ax, title=title,
                           xlabel=lists_union(xlabels),
                           ylabel=lists_union(ylabels),
                           xticks=lists_union(xticks),
                           fontsize=20)
        
        if save:
            self.save_plot(path=self.plots_path, plot_name=f'{indicator}_' + ('_').join(regions))


    def plot_data_distribution(self, data, regions, colors, save=False):
        sns.set_context('paper', rc={'figure.figsize': (16, 8),
                        'legend.title_fontsize': 16, 'legend.fontsize': 14,
                        'figure.titlesize': 20, 'axes.titlesize': 20, 'axes.labelsize': 16,
                        'xtick.labelsize': 14, 'ytick.labelsize': 14})


        grid = sns.pairplot(data, hue='region', palette={region: color for region, color in zip(regions, colors)},
                            kind="scatter", diag_kind="kde", corner=False,
                            plot_kws={'edgecolor': 'black'},
                            diag_kws={'linewidth': 2})
        
        for ax, label in zip(grid.axes[:, 0], [column_name.split('_')[0] for column_name in data.columns if column_name != 'region']):
            ax.set_ylabel(label)
        for ax, label in zip(grid.axes[-1, :], [column_name.split('_')[0] for column_name in data.columns if column_name != 'region']):
            ax.set_xlabel(label)

        if len(regions) == 1:
            grid.fig.subplots_adjust(top=0.9)
            grid.fig.suptitle(regions[0], verticalalignment='top', fontsize=20)
            grid._legend.remove()
        else:
            grid.fig.subplots_adjust(top=0.9)
            grid.fig.suptitle('Data Distribution', verticalalignment='top', fontsize=20)

        if save:
            self.save_plot(path=self.plots_path, plot_name='data_distribution_' + ('_').join(regions))
    
    @staticmethod
    def _get_xticks(values, threshold=0.3):
        sorted_values = sorted(values)

        ticks = [sorted_values[0]]
        for value in sorted_values:
            if value - ticks[-1] >= threshold:
                ticks.append(value)
        ticks[-1] = sorted_values[-1]

        return ticks

    def plot_prediction(self, region, prediction, target, color, save=False):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))

        idel_line_x = np.linspace(np.min(target), np.max(target), 100)
        ax.plot(idel_line_x, idel_line_x, label='target line',
                **{'color': 'red', 'alpha': 0.8, 'linewidth': 1, 'linestyle': '--'})
        ax.scatter(target, target, label='target',
                   **{'color': 'red', 'alpha': 1.0, 'marker': 'o', 'edgecolors': 'black'})

        ax.scatter(target, prediction, label='prediction',
                   **{'color': color, 'alpha': 1.0, 'marker': 'o', 'edgecolors': 'black'})
        
        self.set_ax_params(ax, title=region,
                           xlabel='target',
                           ylabel='prediction',
                           xticks=self._get_xticks(target.to_numpy().ravel()),
                           fontsize=20)
        ax.tick_params(axis='x', rotation=60)
        
        if save:
            self.save_plot(path=self.plots_path, plot_name=f'prediction_{region}')
    
    @staticmethod
    def _compact_feature_names(feature_names):
        compact_names = []
        for feature_name in feature_names:
            compact_names.append(' * '.join([name.split('_')[0] for name in feature_name.split(' ')]))
            if feature_name[-2] == '^' and compact_names[-1][-2] != '^':
                compact_names[-1] += feature_name[-2:]

        return compact_names

    def plot_coefficients_distribution(self, region, regressor, color, save=False):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))

        coefficients_table = pd.DataFrame(data=regressor.model.coef_,
                                          columns=['coefficient'],
                                          index=self._compact_feature_names(regressor.model.feature_names_in_))

        ax = coefficients_table.plot.barh(ax=ax, color=color, fontsize=16)

        ax.set_title('Coefficients Distribution', fontsize=20)  
        ax.legend(loc='best', prop={'size': 14})
        ax.grid(True, alpha=0.5)
        
        if save:
            self.save_plot(path=self.plots_path, plot_name=f'coefficients_distribution_{region}')
    
    def plot_features_std(self, region, features_table, color, save=False):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))

        importance_table = pd.DataFrame(data=features_table.to_numpy(),
                                        columns=self._compact_feature_names(list(features_table.columns)))

        ax = importance_table.std(axis=0).plot.barh(ax=ax, color=color, fontsize=16)

        ax.set_title('Features Standard Deviations', fontsize=20)
        ax.grid(True, alpha=0.5)
        
        if save:
            self.save_plot(path=self.plots_path, plot_name=f'features_std_{region}')
    
    def plot_coefficients_importance(self, region, regressor, features_table, color, save=False):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
        
        importance_table = pd.DataFrame(data=features_table.to_numpy(),
                                        columns=self._compact_feature_names(regressor.model.feature_names_in_))
        
        coefficients_table = pd.DataFrame(data=regressor.model.coef_ * importance_table.std(axis=0),
                                          columns=['importance'],
                                          index=self._compact_feature_names(regressor.model.feature_names_in_))

        ax = coefficients_table.plot.barh(ax=ax, color=color, fontsize=16)

        ax.set_title('Coefficients Importance', fontsize=20)  
        ax.legend(loc='best', prop={'size': 14})
        ax.grid(True, alpha=0.5)
        
        if save:
            self.save_plot(path=self.plots_path, plot_name=f'coefficients_importance_{region}')
    
    def plot_coefficients(self, region, regressor, features_table, color, save=False):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
        
        coefficients_table = pd.DataFrame(data=regressor.model.coef_,
                                          columns=['coefficient'],
                                          index=self._compact_feature_names(regressor.model.feature_names_in_))
        
        importance_table = pd.DataFrame(data=features_table.to_numpy(),
                                        columns=self._compact_feature_names(regressor.model.feature_names_in_))
        
        coefficients_table['importance'] = regressor.model.coef_ * importance_table.std(axis=0)

        ax = coefficients_table.plot.barh(ax=ax, color=color, fontsize=16)

        ax.set_title('Coefficients', fontsize=20)
        ax.legend(loc='best', prop={'size': 14})
        ax.grid(True, alpha=0.5)
        
        if save:
            self.save_plot(path=self.plots_path, plot_name=f'coefficients_{region}')
    
    def plot_UBI_implementation_effect(self, features_table, target, predictor, payment_in_minimum_wage_list, color, save=False):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))

        predictions = []

        for payment_in_minimum_wage in payment_in_minimum_wage_list:
            predictions.append(predictor.predict(features_table=features_table,
                                                 payment_in_minimum_wage=payment_in_minimum_wage))

        ax.hlines(y=target, label=f'real {target.columns[0]}',
                  xmin=np.min(payment_in_minimum_wage_list), xmax=np.max(payment_in_minimum_wage_list),
                  **{'color': 'red', 'alpha': 0.8, 'linewidth': 2, 'linestyle': '--'})

        erroneous_target = predictor.predict(features_table=features_table,
                                             payment_in_minimum_wage=0.0)

        ax.hlines(y=erroneous_target, label=f'erroneous {target.columns[0]}',
                  xmin=np.min(payment_in_minimum_wage_list), xmax=np.max(payment_in_minimum_wage_list),
                  **{'color': 'black', 'alpha': 0.8, 'linewidth': 2, 'linestyle': '--'})

        ax.plot(payment_in_minimum_wage_list, predictions, label=f'predicted {target.columns[0]}',
                **{'color': color, 'alpha': 1.0,
                   'linewidth': 2, 'linestyle': '-',
                   'marker': 'o',
                   'markerfacecolor': color, 'markeredgecolor': 'black'})
        
        self.set_ax_params(ax, title='UBI Implementation Effect',
                           xlabel='payment in minimum wage',
                           ylabel=target.columns[0],
                           xticks=payment_in_minimum_wage_list,
                           fontsize=20)
        ax.tick_params(axis='x', rotation=0)
        
        if save:
            self.save_plot(path=self.plots_path, plot_name=f'UBI_implementation_effect')
