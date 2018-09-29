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
    
    def __lean_and_mean_data_frame(self, data):
        ldf = pd.DataFrame(
                data=data, columns=['Adj Close'], index=data.index
                )
        # rename adjusted close column name as close
        ldf.rename(columns={'Adj Close': 'Close'}, inplace=True)
        return ldf
    
    def get_fin_data_from_yahoo(self, scrip, sd, ed, save=False):
        Validator.validate_attribute(scrip, str, True)
        Validator.validate_attribute(sd, str, True)
        Validator.validate_attribute(ed, str, True)
        data = yf.download(
                tickers=scrip, start=sd, end=ed
                )
        data.to_csv('findata.csv', index=True, encoding='utf-8')
        return self.__lean_and_mean_data_frame(data)
    
    def get_fin_data_from_local(self):
        data = pd.read_csv('findata.csv', parse_dates=True, index_col=0)
        return self.__lean_and_mean_data_frame(data)
    
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
            y.plot(ax=ts_ax, lineWidth=1)
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
        self.tsplot(x, lags=self.__n_lags, saveas='random_walk.png')
    
    def analyse_ts(self, data):
        self.tsplot(
                data.Close,
                lags=self.__n_lags,
                saveas='ts.png'
                )
    
    def analyse_ts_first_diffs(self, data):
        self.tsplot(
                np.diff(data.Close),
                lags=self.__n_lags,
                saveas='ts_first_diffs.png'
                )
    
    def analyse_linear_model(self):
        # y[t] = B0 + B1*t + w[t]
        w = np.random.normal(size=self.__n_samples)
        y = np.empty_like(w)
        B0 = -5000.
        B1 = 25.
        for t in range(len(w)):
            y[t] = B0 + B1*t + w[t]
        self.tsplot(y, lags=self.__n_lags, saveas='linear.png')
    
    def analyse_exponential_model(self):
        '''
        http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
        '''
        # a series of imaginary dates
        dates = pd.date_range('2015-01-01', '2018-09-01', freq='SM')
        # a series of exponentially increasing returns
        sales = [np.exp(x/12) for x in range(1, len(dates) + 1)]
        self.tsplot(sales, lags=self.__n_lags, saveas='exponential.png')
    
    def analyse_log_linear_model(self):
        # a series of imaginary dates
        dates = pd.date_range('2015-01-01', '2018-09-01', freq='SM')
        # a series of exponentially increasing returns
        sales = [np.exp(x/12) for x in range(1, len(dates) + 1)]
        self.tsplot(np.log(sales), lags=self.__n_lags, saveas='log_linear.png')
    
    def analyse_ar_1_with_root(self, a=1.0):
        assert a != 0.0, 'AR root can not be zero'
        assert a < 1.0, 'AR root can not be greater than one'
        np.random.seed(self.__seed)
        x = w = np.random.normal(size=self.__n_samples)
        for t in range(1, self.__n_samples):
            x[t] = a * x[t-1] + w[t]
        self.tsplot(x, lags=self.__n_lags, saveas='ar1_{:04.2f}.png'.format(a))
        # our simulated AR model has order = 1 with alpha = 0.6
        # if we fit an AR(p) model to the above simulated data and ask it to
        # select the order, the selected values of p and a should match with
        # the actual ones
        log('Fitting the AR model to the simulated data...')
        mdl = smt.AR(x).fit(
                maxlag=self.__n_lags,
                method='mle',
                ic='bic',
                trend='nc'
                )
        logn('[Done]')
        log('Estimating the order of the AR model...')
        est_order = smt.AR(x).select_order(
                maxlag=self.__n_lags,
                method='mle',
                ic='bic',
                trend='nc'
                )
        logn('[Done]')
        true_order = 1
        print('alpha estimate: {:3.2f} | best lag order = {}'
              .format(mdl.params[0], est_order))
        print('true alpha = {:3.2f} | true order = {}'
              .format(a, true_order))
    
# %%        
if __name__ == '__main__':
    sc = SerialCorrelation()
    #data = sc.get_fin_data_from_yahoo('TCS', '2016-09-24', '2018-09-24')
#    data = sc.get_fin_data_from_local()
#    sc.serial_correlation(data)
#    sc.analyse_white_noise()
#    sc.analyse_random_walk()
#    sc.analyse_ts(data)
#    sc.analyse_ts_first_diffs(data)
#    sc.analyse_linear_model()
#    sc.analyse_exponential_model()
#    sc.analyse_log_linear_model()
    sc.analyse_ar_1_with_root(a=0.6)
