#!/usr/local/bin/python
# -*- coding: utf-8 -*-

"""
Created on  : Sun Aug  5 17:32:40 2018
@author     : Sourabh
"""

# %%

"""

Sample requests:

https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI?formatted=true&
crumb=4D5ubVRDG3o&lang=en-IN&region=IN&range=1y&interval=1d&events=div%7Csplit&
corsDomain=in.finance.yahoo.com

https://query2.finance.yahoo.com/v8/finance/chart/HDFCBANK.NS?formatted=true&
crumb=4D5ubVRDG3o&lang=en-IN&region=IN&period1=1470335400&period2=1533407400&
interval=1d&events=div%7Csplit&corsDomain=in.finance.yahoo.com

https://query2.finance.yahoo.com/v8/finance/chart/HDFCBANK.NS?formatted=true&
crumb=4D5ubVRDG3o&lang=en-IN&region=IN&period1=1470335400&period2=1533407400&
interval=1d&events=div%7Csplit&corsDomain=in.finance.yahoo.com

Parameters:

formatted=true
crumb=4D5ubVRDG3o
lang=en-IN
region=IN
    range=1y
    or
    period1=1470335400
    period2=1533407400
interval=1d
events=div%7Csplit
corsDomain=in.finance.yahoo.com

"""

import sys

# add modules path before importing them
sys.path.insert(0, '../../')

import DateTimeEpoch as dte
import requests
from urllib.parse import quote
import json
from pyutils.io import FileManager
import pandas as pd
import matplotlib.pyplot as plt


# user configurable use case specific data
scrip_code = '^NSEI'
start_date = '01-04-2018'
end_date = '05-08-2018'
online_mode = False

if online_mode:
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
    period1 = dte.seconds_from_date(start_date)
    period2 = dte.seconds_from_date(end_date)
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
    
    print('Fetching financial data from:', web_url, sep='\n')
    
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
    
    #print(timestamp, opened, high, low, closed, volume, adjclosed, sep='\n\n')
    
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
    for index in range(len(timestamp) - 1, -1, -1):
        data = []
        data.append(dte.date_from_seconds(timestamp[index]))
        data.append(opened[index])
        data.append(high[index])
        data.append(low[index])
        data.append(closed[index])
        data.append(adjclosed[index])
        data.append(volume[index])
        fin_data.append(data)
    
    # save as a CSV file
    FileManager.writeCSV('findata.csv', headers, fin_data, forced = True)

# read as data frame
fin_data_df = pd.read_csv('findata.csv')

## plot the time series
#timeseries = pd.Series(
#        fin_data_df['Adjusted Close'],
#        index = fin_data_df['Date']
#        )
#timeseries.plot.line()

#y_adj_close = fin_data_df['Adjusted Close']
#x_date = fin_data_df['Date']
#plt.plot(
#        x_date,
#        y_adj_close,
#        color = 'blue',
#        lineWidth = 0.7,
#        label = 'Adjusted Close Prices'
#        )

fin_data_df.plot(
        x = 'Date',
        y = 'Close',
        use_index = True,
        label = 'Close Prices'
        )
plt.title(scrip_code + ' (' + start_date + ' to ' + end_date + ')')
plt.legend()
plt.show()
