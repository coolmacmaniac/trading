#!/usr/local/bin/python
# -*- coding: utf-8 -*-

"""
Created on  : Sat Sep 22 17:09:05 2018
@author     : Sourabh
"""

# %%

import sys

import time

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import fix_yahoo_finance as yf
#import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
#import statsmodels.api as sm
#import scipy.stats as scs

from enum import Enum

sys.path.insert(0, '../../')

from pyutils.validations import Validator
from pyutils.visual import Grapher
from pyutils.logger import log, logn

sys.path.remove('../../')


# %%

class SerialCorrelation:
    
    class ModelType(Enum):
        ar = 1
        ma = 2
        arma = 3
    
    def __init__(self):
        self.__n_samples = 1000
        self.__n_lags = 30
        self.g = Grapher()
        pass
    
    @property
    def seed(self):
        '''
        Returns a random seed as integer for next random number generation
        
        :return: a random integer seed
        '''
        rand_seed = int(time.time() % 1000.0)
        return rand_seed
    
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
    
    def get_diminishing_random_list(self, size):
        '''
        Returns a list of given size containing uniformly distributed
        random numbers in descending order
        
        :size: the intended number of elements in the list
        :return: a list of random numbers in descending order
        '''
        np.random.seed(self.seed)
        vals = np.random.uniform(low=0, high=1, size=size)
        vals[::-1].sort()
        return vals
    
    def get_alt_diminishing_random_list(self, size):
        '''
        Returns a list of given size containing uniformly distributed
        random numbers in descending order which are also alternately
        positive and negative
        
        :size: the intended number of elements in the list
        :return: a list of random numbers in descending order alternately
            positive and negative
        '''
        vals = self.get_diminishing_random_list(size)
        for i in range(len(vals)):
            if i % 2 is not 0:
                vals[i] = -vals[i]
        return vals
    
    def get_sample_data(self, m=ModelType.ar, p=1, q=1, n=None):
        Validator.validate_attribute(p, int, True)
        Validator.validate_attribute(q, int, True)
        if n is None:
            n = self.__n_samples
        # the betas of the MA equal to 0 for an AR(p) model likewise
        # the alphas of the AR equal to 0 for an MA(q) model
        if m is SerialCorrelation.ModelType.ar:
            alphas = self.get_alt_diminishing_random_list(size=p)
            betas = np.array([0.])   # value of q does not matter in this case
        elif m is SerialCorrelation.ModelType.ma:
            alphas = np.array([0.])  # value of p does not matter in this case
            betas = self.get_diminishing_random_list(size=q)
        elif m is SerialCorrelation.ModelType.arma:
            alphas = self.get_diminishing_random_list(size=p)
            betas = self.get_diminishing_random_list(size=q)
        # Python requires the zero-lag value as well which is 1
        # also the alphas for the AR model must be negated
        ar = np.r_[1, -alphas]
        ma = np.r_[1, betas]
        data = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n)
        return alphas, betas, data
    
    def fit_ar_model_and_estimate_order(
            self, data, maxlag=None, method='mle', ic='bic', trend='nc'
            ):
        if maxlag is None:
            maxlag = self.__n_lags
        log('Fitting the AR model to the simulated data...')
        mdl = smt.AR(data).fit(
                maxlag=maxlag,
                method=method,
                ic=ic,
                trend=trend
                )
        logn('[Done]')
        log('Estimating the order of the AR model...')
        est_order = smt.AR(data).select_order(
                maxlag=maxlag,
                method=method,
                ic=ic,
                trend=trend
                )
        logn('[Done]')
        return mdl.params, est_order
    
    def analyse_serial_correlation(self, data):
        sc.g.scatplot(data)
    
    def analyse_white_noise(self):
        np.random.seed(self.seed)
        randser = np.random.normal(size=self.__n_samples)
        self.g.tsplot(
                randser,
                lags=self.__n_lags,
                saveas='white_noise.png'
                )
    
    def analyse_random_walk(self):
        np.random.seed(self.seed)
        x = w = np.random.normal(size=self.__n_samples)
        for t in range(1, self.__n_samples):
            x[t] = x[t-1] + w[t]
        self.g.tsplot(x, lags=self.__n_lags, saveas='random_walk.png')
    
    def analyse_ts(self, data):
        self.g.tsplot(
                data.Close,
                lags=self.__n_lags,
                saveas='ts.png'
                )
    
    def analyse_ts_first_diffs(self, data):
        self.g.tsplot(
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
        self.g.tsplot(y, lags=self.__n_lags, saveas='linear.png')
    
    def analyse_exponential_model(self):
        '''
        http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
        '''
        # a series of imaginary dates
        dates = pd.date_range('2015-01-01', '2018-09-01', freq='SM')
        # a series of exponentially increasing returns
        sales = [np.exp(x/12) for x in range(1, len(dates) + 1)]
        self.g.tsplot(sales, lags=self.__n_lags, saveas='exponential.png')
    
    def analyse_log_linear_model(self):
        # a series of imaginary dates
        dates = pd.date_range('2015-01-01', '2018-09-01', freq='SM')
        # a series of exponentially increasing returns
        sales = [np.exp(x/12) for x in range(1, len(dates) + 1)]
        self.g.tsplot(
                np.log(sales),
                lags=self.__n_lags,
                saveas='log_linear.png'
                )
    
    def analyse_ar_1_with_root(self, a=1.0):
        assert a != 0.0, 'AR root can not be zero'
        assert a < 1.0, 'AR root can not be greater than one'
        np.random.seed(self.seed)
        x = w = np.random.normal(size=self.__n_samples)
        for t in range(1, self.__n_samples):
            x[t] = a * x[t-1] + w[t]
        self.g.tsplot(
                x,
                lags=self.__n_lags,
                saveas='ar1_{:04.2f}.png'.format(a)
                )
        # our simulated AR model has order = 1 with alpha = 0.6
        # if we fit an AR(p) model to the above simulated data and ask it to
        # select the order, the selected values of p and a should match with
        # the actual ones
        params, order = self.fit_ar_model_and_estimate_order(x)
        true_order = 1
        logn('alpha estimate: {:3.2f} | best lag order = {}'
              .format(params[0], order))
        logn('true alpha = {:3.2f} | true order = {}'
              .format(a, true_order))
    
    def analyse_ar_p(self, p=1):
        a, b, rts = self.get_sample_data(
                m=SerialCorrelation.ModelType.ar, p=p
                )
        self.g.tsplot(rts,
                      lags=self.__n_lags,
                      saveas='ar{}.png'.format(p)
                      )
        logn('AIC', '='*20, sep='\n')
        params, order = self.fit_ar_model_and_estimate_order(
                rts, maxlag=10, method='mle', ic='aic', trend='nc'
                )
        true_order = p
        logn('alpha estimate: {} | best lag order = {}'
              .format(params, order))
        logn('true alphas = {} | true order = {}'
              .format(a, true_order))
        logn()
        logn('BIC', '='*20, sep='\n')
        params, order = self.fit_ar_model_and_estimate_order(
                rts, maxlag=10, method='mle', ic='bic', trend='nc'
                )
        true_order = p
        logn('alpha estimate: {} | best lag order = {}'
              .format(params, order))
        logn('true alphas = {} | true order = {}'
              .format(a, true_order))
    
    
# %%
if __name__ == '__main__':
    sc = SerialCorrelation()
#    data = sc.get_fin_data_from_yahoo('TCS', '2016-09-24', '2018-09-24')
#    data = sc.get_fin_data_from_local()
#    sc.analyse_serial_correlation(data)
#    sc.analyse_white_noise()
#    sc.analyse_random_walk()
#    sc.analyse_ts(data)
#    sc.analyse_ts_first_diffs(data)
#    sc.analyse_linear_model()
#    sc.analyse_exponential_model()
#    sc.analyse_log_linear_model()
#    sc.analyse_ar_1_with_root(a=0.6)
    sc.analyse_ar_p(p=3)
