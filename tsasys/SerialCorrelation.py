#!/usr/local/bin/python
# -*- coding: utf-8 -*-

"""
Created on  : Sat Sep 22 17:09:05 2018
@author     : Sourabh
"""

# %%

import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fix_yahoo_finance as yf
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

sys.path.insert(0, '../../')

from pyutils.validations import Validator
from pyutils.logger import log, logn

sys.path.remove('../../')


# %%

class SerialCorrelation:
    
    def __init__(self):
        self.__seed = 666
        self.__n_samples = 1000
        self.__n_lags = 30
        pass
    
    def get_fin_data_from_yahoo(self, scrip, sd, ed, save=False):
        Validator.validate_attribute(scrip, str, True)
        Validator.validate_attribute(sd, str, True)
        Validator.validate_attribute(ed, str, True)
        data = yf.download(
                tickers=scrip, start=sd, end=ed
                )
        data.to_csv('findata.csv', index=True, encoding='utf-8')
        return data
    
    def get_fin_data_from_local(self):
        data = pd.read_csv('findata.csv', parse_dates=True, index_col=0)
        return data
    
    def serial_correlation(self, data):
        # scatterplot grid config
        ncols = 3
        nrows = 3
        lags = ncols * nrows
        
        # build scatterplot grid
        fig, axes = plt.subplots(
                ncols=ncols, nrows=nrows, figsize=(7 * ncols, 2.5 * nrows)
                )
        
        # plot scatterplot in each of the box on the grid
        for axis, lag in zip(axes.flat, np.arange(1, lags + 1, 1)):
            lag_str = 't-{}'.format(lag)
            X = pd.concat(
                    objs=[data.Close, data.Close.shift(-lag)],
                    axis=1,
                    keys=['y']+[lag_str]
                    ).dropna()
            
            # plot the data
            axis.scatter(X[lag_str], X['y'], s=1, c='#DD00AA')
            #X.plot(ax=axis, kind='scatter', y='y', x=lag_str, c='#DD00AA')
            corr = X.corr().values[0][1]
            axis.set_xlabel('')
            axis.set_ylabel('Original')
            axis.set_title('Lag: {} (corr={:.2f})'.format(lag_str, corr))
            axis.set_aspect(9/16)
        
        fig.tight_layout()
        plt.savefig('serial_correlation.png')
        plt.show()
    
    def matplot_config_params(self):
        print(
            plt.rcParams['font.family'],
            plt.rcParams['font.sans-serif'],
            plt.rcParams['font.serif'],
            plt.rcParams['font.monospace'],
            plt.rcParams['font.size'],
            plt.rcParams['axes.labelsize'],
            plt.rcParams['axes.labelweight'],
            plt.rcParams['xtick.labelsize'],
            plt.rcParams['ytick.labelsize'],
            plt.rcParams['legend.fontsize'],
            plt.rcParams['figure.titlesize'],
            sep='\n'
        )

    def tsplot(self, y, lags=None, figsize=(16, 8), style='bmh', saveas=None):
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        with plt.style.context(style):
            plt.figure(figsize=figsize)
            layout = (3, 2)
            ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
            acf_ax = plt.subplot2grid(layout, (1, 0))
            pacf_ax = plt.subplot2grid(layout, (1, 1))
            qq_ax = plt.subplot2grid(layout, (2, 0))
            pp_ax = plt.subplot2grid(layout, (2, 1))
            y.plot(ax=ts_ax)
            ts_ax.set_title('Time Series Analysis Plots')
            smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
            smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
            sm.qqplot(y, line='s', ax=qq_ax)
            qq_ax.set_title('QQ Plot')
            scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)
            plt.tight_layout()
            if saveas is not None and isinstance(saveas, str):
                plt.savefig(saveas)
    
    def analyse_white_noise(self):
        np.random.seed(self.__seed)
        randser = np.random.normal(size=self.__n_samples)
        self.tsplot(
                randser,
                lags=self.__n_lags,
                saveas='white_noise.png'
                )
    
    def analyse_random_walk(self):
        np.random.seed(self.__seed)
        x = w = np.random.normal(size=self.__n_samples)
        for t in range(1, self.__n_samples):
            x[t] = x[t-1] + w[t]
        self.tsplot(x, lags=30, saveas='random_walk.png')

# %%
if __name__ == '__main__':
    sc = SerialCorrelation()
    #data = sc.get_fin_data_from_yahoo('TCS', '2016-09-24', '2018-09-24')
    data = sc.get_fin_data_from_local()
    sc.serial_correlation(data)
    sc.analyse_white_noise()
    sc.analyse_random_walk()