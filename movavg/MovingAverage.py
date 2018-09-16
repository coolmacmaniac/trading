#!/usr/local/bin/python
# -*- coding: utf-8 -*-

"""
Created on  : Sun Sep 16 19:20:37 2018
@author     : Sourabh
"""

# %%

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc

sys.path.insert(0, '../../')

from trading.findata import Historical
from pyutils.datetime import DateTime
from pyutils.validations import Validator

sys.path.remove('../../')


class MovingAverage(Historical):
    
    def __init__(self, go_online=False, do_save=False):
        super().__init__(go_online, do_save)
        pass
    
    def plot_ma(self, date, movavg, lbl):
        plt.plot(
                date,
                movavg,
                label=lbl,
                lineWidth=0.7
                )
    
    def plot_params(self, df, recent):
        return super().plot_params(df, recent)
    
    def show_and_save(self, plt, show, save, fname):
        super().show_and_save(plt, show, save, fname)
    
    def compute_moving_averages(self, df):
        dohlc = df.copy()
        col = 'Adjusted Close'
        dohlc['SMA 5'] = dohlc[col].rolling(5).mean()
        dohlc['SMA 10'] = dohlc[col].rolling(10).mean()
        dohlc['SMA 15'] = dohlc[col].rolling(15).mean()
        dohlc['SMA 20'] = dohlc[col].rolling(20).mean()
        dohlc['SMA 50'] = dohlc[col].rolling(50).mean()
        dohlc['SMA 100'] = dohlc[col].rolling(100).mean()
        dohlc['SMA 150'] = dohlc[col].rolling(150).mean()
        dohlc['SMA 200'] = dohlc[col].rolling(200).mean()
        dohlc['SMA 250'] = dohlc[col].rolling(250).mean()
        dohlc = dohlc.dropna()
        return dohlc
    
    def line_plot_time_series_with_ma(
            self, scrip_code, df, recent=0, show=True, save=True
            ):
        count, steps, dfmt = self.plot_params(df, recent)
        figure, axis = plt.subplots(figsize=(16, 8))
        dohlc = df.tail(count).copy()
        dohlc.plot(
            kind='line',
            y='Adjusted Close',
            label='Closed Prices (INR)',
            lineWidth=0.7,
            ax=axis
        )
        self.plot_ma(dohlc.index, dohlc['SMA 5'], 'SMA 5')
        self.plot_ma(dohlc.index, dohlc['SMA 10'], 'SMA 10')
        self.plot_ma(dohlc.index, dohlc['SMA 15'], 'SMA 15')
        self.plot_ma(dohlc.index, dohlc['SMA 20'], 'SMA 20')
        axis.xaxis.set_major_formatter(mdates.DateFormatter(dfmt))
        axis.set_xticks(dohlc.index[::-steps])
        plt.setp(plt.gca().get_xticklabels(), rotation=90)
        start_date = DateTime.date_time_to_formatted_date(dohlc.index[0])
        end_date = DateTime.date_time_to_formatted_date(dohlc.index[-1])
        plt.title(scrip_code + ' (' + start_date + ' to ' + end_date + ')')
        plt.xlabel('')
        plt.legend()
        plt.tight_layout()
        self.show_and_save(plt, show, save, 'timeseries.png')
    
    def candle_plot_time_series_with_ma(
            self, scrip_code, df, recent=0, show=True, save=True
            ):
        Validator.validate_attribute(recent, int, True)
        count, steps, dfmt = self.plot_params(df, recent)
        figure, axis = plt.subplots(figsize=(16, 8))
        # create a copy of the DataFrame to operate on
        dohlc = df.tail(count).copy()
        # take the index back to column
        dohlc.reset_index(inplace=True)
        # drop the closed price column, adj.closed price will be considered
        dohlc.drop('Close', axis=1, inplace=True)
        # convert the datetime format to short string format
        dohlc.Date = dohlc.Date.apply(DateTime.dateformatter_short)
        # get hold of dates for title naming
        str_dates = dohlc.Date
        # convert the string dates to pandas TimeStamp values
        dohlc.Date = pd.to_datetime(dohlc.Date, format='%d-%m-%Y')
        # convert the pandas TimeStamp values to matplotlib float values
        dohlc.Date = dohlc.Date.apply(mdates.date2num)
        candlestick_ohlc(
                ax=axis,
                quotes=dohlc.values,
                width=0.6,
                colorup='green',
                colordown='red',
                alpha=0.7
                )
        self.plot_ma(dohlc['Date'], dohlc['SMA 5'], 'SMA 5')
        self.plot_ma(dohlc['Date'], dohlc['SMA 10'], 'SMA 10')
        self.plot_ma(dohlc['Date'], dohlc['SMA 15'], 'SMA 15')
        self.plot_ma(dohlc['Date'], dohlc['SMA 20'], 'SMA 20')
        axis.xaxis.set_major_formatter(mdates.DateFormatter(dfmt))
        axis.set_xticks(dohlc.Date[::-steps])
        plt.setp(plt.gca().get_xticklabels(), rotation=90)
        start_date = str_dates.iloc[0]
        end_date = str_dates.iloc[-1]
        plt.title(scrip_code + ' (' + start_date + ' to ' + end_date + ')')
        plt.tight_layout()
        plt.legend()
        self.show_and_save(plt, show, save, 'candlesticks.png')


# %%
if __name__ == '__main__':
    ma = MovingAverage(go_online=False, do_save=False)
    symbol = 'TCS'
    data = ma.get(
            symbol,
            DateTime.five_years_ago_from_today(),
            DateTime.today()
            )
    dohlc = ma.compute_moving_averages(data)
    ma.line_plot_time_series_with_ma(
            symbol, dohlc, recent=60, show=True, save=True
            )
    ma.candle_plot_time_series_with_ma(
            symbol, dohlc, recent=60, show=True, save=True
            )
