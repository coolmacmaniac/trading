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
from urllib.parse import quote

sys.path.insert(0, '../../')

from pyutils.datetime import DateTime
from pyutils.validations import Validator
from pyutils.logger import log, logn

sys.path.remove('../../')


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
    
    def line_plot_time_series(self, scrip_code, df):
        figure, axis = plt.subplots(figsize=(16, 8))
        df.plot(
            kind='line',
            y='Adjusted Close',
            label='Closed Prices (INR)',
            lineWidth=0.7,
            ax=axis
        )
        axis.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
        axis.set_xticks(df.index[::60])
        start_date = DateTime.date_time_to_formatted_date(df.index[0])
        end_date = DateTime.date_time_to_formatted_date(df.index[-1])
        plt.title(scrip_code + ' (' + start_date + ' to ' + end_date + ')')
        plt.xlabel('')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig('timeseries.png')

    def get(self, symbol=None, from_date=None, to_date=None):
        Validator.validate_attribute(symbol, str, True)
        if from_date is None:
            from_date = DateTime.five_years_ago_from_today()
        if to_date is None:
            to_date = DateTime.today()
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
            DateTime.today()
            )
    fh.line_plot_time_series(symbol, data)
