#!/usr/local/bin/python
# -*- coding: utf-8 -*-

"""
Created on  : Mon Aug  6 23:25:10 2018
@author     : Sourabh
"""

# %%

from datetime import datetime
import re

fmt_date_time = '%d-%m-%Y %H:%M:%S'
fmt_date = '%d-%m-%Y'
re_fmt_date_time = '(\d+)-(\d+)-(\d+) (\d+):(\d+):(\d+)'
re_fmt_date = '(\d+)-(\d+)-(\d+)'

def formatted_string_from_seconds(format_specifier, seconds):
    if isinstance(seconds, str):
        seconds = int(seconds)
    printable_str = datetime.fromtimestamp(seconds).strftime(format_specifier)
    return printable_str

def components_from_date_string(date_str):
    match = re.match(r'{}'.format(re_fmt_date), date_str)
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))

def components_from_date_time_string(date_str):
    match = re.match(r'{}'.format(re_fmt_date_time), date_str)
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)),
            int(match.group(4)), int(match.group(5)), int(match.group(6)))

def date_time_from_seconds(seconds):
    
    return formatted_string_from_seconds(fmt_date_time, seconds)

def date_from_seconds(seconds):
    return formatted_string_from_seconds(fmt_date, seconds)

def seconds_from_date(date_str):
    (d, m, Y) = components_from_date_string(date_str)
    seconds = datetime(Y, m, d).strftime('%s')
    return seconds

def seconds_from_date_time(date_time_str):
    (d, m, Y, H, M, S) = components_from_date_time_string(date_time_str)
    seconds = datetime(Y, m, d, H, M, S).strftime('%s')
    return seconds

#ts1 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(1190000700))
#ts2 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(1533267900))
#ts3 = date_time_from_seconds(1533181500)

#ts1 = date_from_seconds(1190000700)        # 1189967400
#ts2 = date_time_from_seconds(1190000700)
#ts3 = seconds_from_date(ts1)
#ts4 = seconds_from_date_time(ts2)

#print(type(ts1), ts1)
#print(type(ts2), ts2)
#print(type(ts3), ts3)
#print(type(ts4), ts4)
