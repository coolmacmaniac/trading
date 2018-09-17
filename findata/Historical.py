#!/usr/local/bin/python
# -*- coding: utf-8 -*-

"""
Created on  : Sat Sep 15 13:57:35 2018
@author     : Sourabh
"""

# %%

import sys

import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc
from urllib.parse import quote

sys.path.insert(0, '../../')

from pyutils.datetime import DateTime
from pyutils.validations import Validator
from pyutils.logger import log, logn

sys.path.remove('../../')


# %%

#import fix_yahoo_finance as yf
#data = yf.download("^NSEI", start="2018-09-01", end="2018-09-16")
#data.tail()


# %%
class Historical:
    
    def __init__(self, go_online=False, do_save=False):
        self.__online_mode = go_online
        self.__save_mode = do_save
        self.__csv_path = 'histdata.csv'
    
    def __construct_web_url(self, scrip_code, start_date, end_date):
        # components of the URL
        protocol = 'https://'
        host_name = 'query2.finance.yahoo.com'
        path_fmt = '/v8/finance/chart/{}'
        params_fmt = '?formatted={}&crumb={}&lang={}&region={}&period1={}&' + \
                        'period2={}&interval={}&events={}&corsDomain={}'
        # constant initializations needed in the URL
        formatted = True
        crumb = '4D5ubVRDG3o'
        lang = 'en-IN'
        region = 'IN'
        period1 = DateTime.seconds_from_date(start_date)
        period2 = DateTime.seconds_from_date(end_date)
        interval = '1d'
        events = 'div%7Csplit'
        corsDomain = 'in.finance.yahoo.com'
        # populate the path and other parameters using the computed values
        filled_path = path_fmt.format(quote(scrip_code))
        filled_params = params_fmt.format(
                formatted,
                crumb,
                lang,
                region,
                period1,
                period2,
                interval,
                events,
                corsDomain
                )
        # constructing the complete url for GET request
        web_url = protocol + host_name + filled_path + filled_params
        logn('='*40)
        logn(web_url)
        logn('-'*20)
        logn('Looking for ({}) from {} till {}...[Done]'.format(
                scrip_code, start_date, end_date
                )
        )
        return web_url
    
    def __fetch_and_parse_json(self, web_url: str):
        log('Fetching financial data from Yahoo Finance...')
        # query the web server at URL and return the JSON response
        web_request = requests.get(web_url)
        web_response = web_request.text
        # get hold of respective fields
        json_obj = json.loads(web_response)
        timestamp = json_obj['chart']['result'][0]['timestamp']
        indicators = json_obj['chart']['result'][0]['indicators']
        opened = indicators['quote'][0]['open']
        high = indicators['quote'][0]['high']
        low = indicators['quote'][0]['low']
        closed = indicators['quote'][0]['close']
        volume = indicators['quote'][0]['volume']
        adjclosed = indicators['adjclose'][0]['adjclose']
        logn('[Done]')
        log('Parsing the json response...')
        fin_data = []
        headers = [
                'Date',
                'Open',
                'High',
                'Low',
                'Close',
                'Adjusted Close',
                'Volume'
                ]
        # extract information of each field and keep in a list of lists
        for index in range(len(timestamp)):
            if opened[index] is None or high[index] is None or \
                low[index] is None or closed[index] is None or \
                volume[index] is None or adjclosed[index] is None:
                    continue
            data = []
            data.append(DateTime.date_string_from_seconds(timestamp[index]))
            data.append(opened[index])
            data.append(high[index])
            data.append(low[index])
            data.append(closed[index])
            data.append(adjclosed[index])
            data.append(volume[index])
            fin_data.append(data)
        df = pd.DataFrame(fin_data, columns=headers)
        # replacing zeros with respective avg so that they can be handled later
        df = df[headers].replace({'0': np.nan, 0: np.nan})
        # ignore date column for mean calculations
        headers = headers[1:]
        # replace NaN in each column with respective column mean
        for header in headers:
            df[header].fillna(df[header].mean(), inplace=True)
        logn('[Done]')
        if self.__save_mode:
            log('Exporting the data to CSV file...')
            # save as a CSV file
            csv_path = self.__csv_path
            df.to_csv(csv_path, index=False, encoding='utf-8')
            logn('[Done]')
            logn('Saved file: {}'.format(csv_path))
        # parse Date column as python datetime
        df.Date = df.Date.apply(DateTime.dateparser_short)
        # re-index the dataframe on converted Date column
        df.set_index('Date', drop=True, inplace=True)
        return df
    
    def plot_params(self, df, recent):
        if recent is 0:
            count = df.index.size
            steps = 20
            dfmt = '%b-%Y'
        else:
            count = recent
            dfmt = '%d-%b-%Y'
            if recent > 90:
                dfmt = '%b-%Y'
                steps = 20
            elif recent > 60:
                steps = 5
            else:
                steps = 1
        return count, steps, dfmt
    
    def show_and_save(self, plt, show, save, fname):
        if show:
            plt.show()
        if not show and save:
            plt.savefig(fname)
            plt.clf()
            plt.close()
        if show and save:
            plt.savefig(fname)
        if not show and not save:
            plt.clf()
            plt.close()
    
    def line_plot_time_series(
            self, scrip_code, df, recent=0, show=True, save=True
            ):
        count, steps, dfmt = self.__plot_params(df, recent)
        figure, axis = plt.subplots(figsize=(16, 8))
        dohlc = df.tail(count).copy()
        dohlc.plot(
            kind='line',
            y='Adjusted Close',
            label='Closed Prices (INR)',
            lineWidth=0.7,
            ax=axis
        )
        axis.xaxis.set_major_formatter(mdates.DateFormatter(dfmt))
        axis.set_xticks(dohlc.index[::-steps])
        plt.setp(plt.gca().get_xticklabels(), rotation=90)
        start_date = DateTime.date_time_to_formatted_date(dohlc.index[0])
        end_date = DateTime.date_time_to_formatted_date(dohlc.index[-1])
        plt.title(scrip_code + ' (' + start_date + ' to ' + end_date + ')')
        plt.xlabel('')
        plt.legend()
        plt.tight_layout()
        self.__show_and_save(plt, show, save, 'timeseries.png')
    
    def candle_plot_time_series(
            self, scrip_code, df, recent=0, show=True, save=True
            ):
        Validator.validate_attribute(recent, int, True)
        count, steps, dfmt = self.__plot_params(df, recent)
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
        axis.xaxis.set_major_formatter(mdates.DateFormatter(dfmt))
        axis.set_xticks(dohlc.Date[::-steps])
        plt.setp(plt.gca().get_xticklabels(), rotation=90)
        start_date = str_dates.iloc[0]
        end_date = str_dates.iloc[-1]
        plt.title(scrip_code + ' (' + start_date + ' to ' + end_date + ')')
        plt.tight_layout()
        self.__show_and_save(plt, show, save, 'candlesticks.png')
        
    
    def get(self, symbol=None, from_date=None, to_date=None):
        Validator.validate_attribute(symbol, str, True)
        if from_date is None:
            from_date = DateTime.five_years_ago_from_today()
        if to_date is None:
            to_date = DateTime.today_extended()
        if self.__online_mode:
            weburl = self.__construct_web_url(symbol, from_date, to_date)
            hist_data = self.__fetch_and_parse_json(weburl)
        else:
            csv_path = self.__csv_path
            hist_data = pd.read_csv(
                    csv_path, index_col='Date',
                    parse_dates=True, date_parser=DateTime.dateparser_short
                    )
        return hist_data


# %%
if __name__ == '__main__':
    fh = Historical(go_online=False, do_save=False)
    symbol = '^NSEI'
    data = fh.get(
            symbol,
            DateTime.five_years_ago_from_today(),
            DateTime.today_extended()
            )
    fh.line_plot_time_series(symbol, data)
    fh.candle_plot_time_series(symbol, data, recent=30)
