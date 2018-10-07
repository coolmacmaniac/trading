#!/usr/local/bin/python
# -*- coding: utf-8 -*-

"""
Created on  : Sat Sep 22 17:09:05 2018
@author     : Sourabh
"""

# %%

import sys

import time
import quandl

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
    
    def __logged_data(self, data):
        # all available columns are converted to logged values
        # after shifted division
        lrets = np.log(data/data.shift(1)).dropna()
        return lrets
        
    def get_fin_data_from_yahoo(self, scrip, sd, ed, save=False):
        Validator.validate_attribute(scrip, str, True)
        Validator.validate_attribute(sd, str, True)
        Validator.validate_attribute(ed, str, True)
        data = yf.download(
                tickers=scrip, start=sd, end=ed
                )
        data.to_csv('findata.csv', index=True, encoding='utf-8')
        return self.__lean_and_mean_data_frame(data)
    
    def get_fin_data_from_quandl(self, scrip, sd, ed, save=False):
        Validator.validate_attribute(scrip, str, True)
        Validator.validate_attribute(sd, str, True)
        Validator.validate_attribute(ed, str, True)
        data = quandl.get(
                api_key='W48T46x32auyA_jwkzXT',
                dataset=scrip,
                start_date=sd,
                end_date=ed
                )
        dropcols = [
                'Last Traded Price',
                'Turnover (in Lakhs)'
                ]
        cols = [
                'Open',
                'High',
                'Low',
                'Adj Close',
                'Volume'
                ]
        data.drop(columns=dropcols, inplace=True)
        data.rename(columns={'Close Price': 'Adj Close'}, inplace=True)
        data.to_csv('fd.csv', index=True, header=cols, encoding='utf-8')
        return self.__lean_and_mean_data_frame(data)
    
    def get_log_ret_data_from_yahoo(self, symbols, sd, ed, save=False):
        if symbols is None:
            raise ValueError(
                    'The attribute "symbols" has to be provided'
                    )
        if isinstance(symbols, str):
            symbols = [symbols]
        Validator.validate_attribute(sd, str, True)
        Validator.validate_attribute(ed, str, True)
        get_px = lambda x: yf.download(
                tickers=x, start=sd, end=ed
                )['Adj Close']
        # raw adjusted close prices
        data = pd.DataFrame({sym:get_px(sym) for sym in symbols})
        # log returns
        lrets = np.log(data/data.shift(1)).dropna()
        lrets.rename(
                columns={'SPY': 'LSPY', 'TLT': 'LTLT', 'MSFT': 'LMSFT'},
                inplace=True
                )
        data = pd.concat([data, lrets], axis=1).dropna()
        if save:
            data.to_csv(
                    'logged_fd.csv', index=True, encoding='utf-8'
                    )
        return data
    
    def get_fin_data_from_local(self, file_name):
        data = pd.read_csv(file_name, parse_dates=True, index_col=0)
        return self.__lean_and_mean_data_frame(data)
    
    def get_log_ret_data_from_local(self, file_name):
        data = pd.read_csv(file_name, parse_dates=True, index_col=0)
        return data
    
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
    
    def get_sample_data(self, m=ModelType.ar, p=1, q=1, n=None, b=0):
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
            #betas = np.array([0.6, 0.4, 0.2])
        elif m is SerialCorrelation.ModelType.arma:
            alphas = self.get_alt_diminishing_random_list(size=p)
            betas = self.get_diminishing_random_list(size=q)
            #alphas = np.array([0.5, -0.25, 0.4])
            #betas = np.array([0.5, -0.3])
        # Python requires the zero-lag value as well which is 1
        # also the alphas for the AR model must be negated
        ar = np.r_[1, -alphas]
        ma = np.r_[1, betas]
        data = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n, burnin=b)
        return alphas, betas, data
    
    def fit_ar_model_and_estimate_order(
            self, data, maxlag=None, method='mle', ic='bic', trend='nc'
            ):
        if maxlag is None:
            maxlag = self.__n_lags
        log('Fitting the AR model to the given data...')
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
    
    def fit_arma_model_and_estimate_order(
            self, data, order=(0, 1), maxlag=None,
            method='mle', trend='nc', burnin=0
            ):
        if maxlag is None:
            maxlag = self.__n_lags
        log('Fitting & estimating the ARMA model to the given data...')
        mdl = smt.ARMA(data, order=order).fit(
                maxlag=maxlag,
                method=method,
                trend=trend,
                burnin=burnin
                )
        logn('[Done]')
        logn(mdl.summary())
        return mdl.arparams, mdl.k_ar, mdl.maparams, mdl.k_ma
    
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
    
    def analyse_ts_log_returns_as_ar_process(self, data):
        logged = self.__logged_data(data)
        self.g.tsplot(
                logged.Close,
                lags=self.__n_lags,
                saveas='ts_log_returns.png'
                )
        logn('BIC', '='*20, sep='\n')
        params, order = self.fit_ar_model_and_estimate_order(
                logged.Close, maxlag=10, method='mle', ic='bic', trend='nc'
                )
        if order is 1:
            logn('alpha estimate: {:.5f} | best lag order = {}'
              .format(params[0], order))
        else:
            logn('alpha estimate: {} | best lag order = {}'
                  .format(params, order))
    
    def analyse_ma_q(self, q=1):
        a, b, rts = self.get_sample_data(
                m=SerialCorrelation.ModelType.ma, q=q
                )
        self.g.tsplot(rts,
                      lags=self.__n_lags,
                      saveas='ma{}.png'.format(q)
                      )
        try:
            _, _, params, order = self.fit_arma_model_and_estimate_order(
                rts, maxlag=10, order=(0, q), method='mle', trend='nc'
                )
            logn('beta estimate: {} | best lag order = {}'
              .format(params, order))
        except ValueError:
            pass
        true_order = q
        logn('true betas = {} | true order = {}'
              .format(b, true_order))
    
    def analyse_arma_p_q(self, p=1, q=1):
        n = self.__n_samples
        burns = n // 10
        a, b, rts = self.get_sample_data(
                m=SerialCorrelation.ModelType.arma, p=p, q=q, n=n, b=burns
                )
        self.g.tsplot(rts,
                      lags=self.__n_lags,
                      saveas='arma{}{}.png'.format(p, q)
                      )
        try:
            ar_p, ar_o, ma_p, ma_o = self.fit_arma_model_and_estimate_order(
                rts, maxlag=10, order=(p, q),
                method='mle', trend='nc', burnin=burns
                )
            logn('alpha estimate: {} | best ar lag order = {}'
              .format(ar_p, ar_o))
            logn('beta estimate: {} | best ma lag order = {}'
              .format(ma_p, ma_o))
        except ValueError:
            pass
        
        logn('true alphas = {} | true ar order = {}'
             .format(a, p))
        logn('true betas = {} | true ma order = {}'
             .format(b, q))
    
    def analyse_arma_p_q_best_ic(self, p=1, q=1):
        n = 5000
        burns = 2000
        a, b, rts = self.get_sample_data(
                m=SerialCorrelation.ModelType.arma, p=p, q=q, n=n, b=burns
                )
        self.g.tsplot(rts,
                      lags=self.__n_lags,
                      saveas='arma{}{}.png'.format(p, q)
                      )
        # pick best order by minimum ic - aic or bic
        # smallest ic value wins
        best_ic = np.inf
        best_order = None
        best_mdl = None
        rng = range(5)
        for i in rng:
            for j in rng:
                try:
                    tmp_mdl = smt.ARMA(rts, order=(i, j)).fit(
                            method='mle', trend='nc'
                            )
                    tmp_ic = tmp_mdl.bic    # using bic here
                    if tmp_ic < best_ic:
                        best_ic = tmp_ic
                        best_order = (i, j)
                        best_mdl = tmp_mdl
                except: continue
        logn(best_mdl.summary())
        logn('using BIC', '='*20, sep='\n')
        logn('true order: ({}, {})'.format(p, q))
        logn('true alphas = {}'.format(a))
        logn('true betas = {}'.format(b))
        logn('ic: {:6.5f} | estimated order: {}'.format(best_ic, best_order))
        logn('estimated alphas = {}'.format(best_mdl.arparams))
        logn('estimated betas = {}'.format(best_mdl.maparams))
        # analysing the model residuals with the estimated information
        # the residuals should be a white noise process with no serial
        # correlation for any lag, if this is the case then we can say
        # that the best model has been fit to explain the data
        self.g.tsplot(best_mdl.resid,
                      lags=self.__n_lags,
                      saveas='arma{}{}_residuals.png'.format(
                              best_order[0], best_order[1]
                              )
                      )
    
    def analyse_ts_arma(self, data):
        ts = self.__logged_data(data).Close
        best_ic = np.inf
        best_order = None
        best_mdl = None
        rng = range(5)      # orders greater than 5 are not practically useful
        for i in rng:
            for j in rng:
                try:
                    tmp_mdl = smt.ARMA(ts, order=(i, j)).fit(
                            method='mle', trend='nc'
                            )
                    tmp_ic = tmp_mdl.bic    # using bic here
                    logn('ic={}, order=({}, {})'.format(tmp_ic, i, j))
                    if tmp_ic < best_ic:
                        best_ic = tmp_ic
                        best_order = (i, j)
                        best_mdl = tmp_mdl
                except: continue
        logn(best_mdl.summary())
        logn('using BIC', '='*20, sep='\n')
        logn('ic: {:6.5f} | estimated order: {}'.format(best_ic, best_order))
        logn('estimated alphas = {}'.format(best_mdl.arparams))
        logn('estimated betas = {}'.format(best_mdl.maparams))
        self.g.tsplot(best_mdl.resid,
                      lags=self.__n_lags,
                      saveas='ts_arma{}{}_residuals.png'.format(
                              best_order[0], best_order[1]
                              )
                      )
    
    def analyse_ts_arima(self, data):
        ts = data.LSPY
        best_ic = np.inf
        best_order = None
        best_mdl = None
        pq_rng = range(5)    # orders greater than 5 are not practically useful
        d_rng = range(2)     # [0,1]
        for i in pq_rng:
            for d in d_rng:
                for j in pq_rng:
                    try:
                        tmp_mdl = smt.ARIMA(ts, order=(i, d, j)).fit(
                                method='mle', trend='nc'
                                )
                        tmp_ic = tmp_mdl.bic    # using bic here
                        logn('ic={}, order=({}, {}, {})'.format(tmp_ic,i,d,j))
                        if tmp_ic < best_ic:
                            best_ic = tmp_ic
                            best_order = (i, d, j)
                            best_mdl = tmp_mdl
                    except: continue
        logn(best_mdl.summary())
        logn('using BIC', '='*20, sep='\n')
        logn('ic: {:6.5f} | estimated order: {}'.format(best_ic, best_order))
        logn('estimated alphas = {}'.format(best_mdl.arparams))
        logn('estimated betas = {}'.format(best_mdl.maparams))
        self.g.tsplot(best_mdl.resid,
                      lags=self.__n_lags,
                      saveas='ts_arima{}{}{}_residuals.png'.format(
                              best_order[0], best_order[1], best_order[2]
                              )
                      )
    
    
# %%
if __name__ == '__main__':
    sc = SerialCorrelation()
#    data = sc.get_fin_data_from_yahoo('TCS', '2016-09-24', '2018-09-24')
#    data = sc.get_fin_data_from_quandl('TC1/TCS', '2000-01-01', '2018-10-06')
#    data = sc.get_log_ret_data_from_yahoo(
#            ['SPY', 'TLT', 'MSFT'], '2000-01-01', '2018-10-07', True
#            )
#    data = sc.get_fin_data_from_local('findata.csv')
#    data = sc.get_fin_data_from_local('fd.csv')
    data = sc.get_log_ret_data_from_local('logged_fd.csv')
#    sc.analyse_serial_correlation(data)
#    sc.analyse_white_noise()
#    sc.analyse_random_walk()
#    sc.analyse_ts(data)
#    sc.analyse_ts_first_diffs(data)
#    sc.analyse_linear_model()
#    sc.analyse_exponential_model()
#    sc.analyse_log_linear_model()
#    sc.analyse_ar_1_with_root(a=0.6)
#    sc.analyse_ar_p(p=3)
#    sc.analyse_ts_log_returns_as_ar_process(data)
#    sc.analyse_ma_q(q=3)
#    sc.analyse_arma_p_q(p=2, q=2)
#    sc.analyse_arma_p_q_best_ic(p=3, q=2)
#    sc.analyse_ts_arma(data)    # data was older and was fetched from quandl
    sc.analyse_ts_arima(data)   # logged data was fetched from yahoo
