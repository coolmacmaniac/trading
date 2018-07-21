# all imports
import sys

# add modules path before importing them
sys.path.insert(0, '../../')

from bs4 import BeautifulSoup
import requests
from pyutils.io import FileManager
import pandas as pd

use_live_scrapping = False

if use_live_scrapping:
    # use the live web page feed for scrapping
    # save the URL of the web page to be scrapped
    web_url = 'https://zerodha.com/margin-calculator/Futures/'

    # query the website and return the html of the page from URL
    web_request = requests.get(web_url)
    web_content = web_request.content
else:
    # use the file contents for scrapping, to avoid unnecessary hits
    fm = FileManager()
    web_content = fm.read('margins.html')

# parse the content usign beautiful soup by assuming it is html format
#soup = BeautifulSoup(web_content, 'html.parser')
soup = BeautifulSoup(web_content, 'html5lib')

# check print
#print(soup.prettify())

# extract the table layer
#table_div = soup.find('div', attrs = { 'class': 'table_container' })
#table_div = soup.find('div', attrs = { 'id': 'header-container' })
#table = soup.find('table', attrs = { 'id': 'table' })
table_layer = soup.find('table', attrs = { 'class': 'data futures' })
table = table_layer.find('tbody')

margins = []
headers = [
        'scrip',
        'expiry',
        'lot_size',
        'price',
        'margin_nrml_pctg',
        'margin_nrml',
        'margin_mis',
        'mwpl_pctg'
        ]

# iterage each row to extract the margins parameters for each contract
# multiple rows could be there for same contract (like near, mid, far)
for row in table.findAll('tr'):
    margin = []
    margin.append(row.get('data-scrip'))
    margin.append(row.get('data-expiry'))
    margin.append(row.get('data-lot_size'))
    margin.append(row.get('data-price'))
    margin.append(row.get('data-margin'))
    margin.append(row.get('data-nrml_margin'))
    margin.append(row.get('data-mis_margin'))
    tag_mwpl = row.find('td', attrs = { 'class': 'mwpl' })
    margin.append(tag_mwpl.text.strip('\n %'))
    margins.append(margin)

# save as a CSV file
fm.writeCSV('margins.csv', headers, margins)

# read as data frame
margins_df = pd.read_csv('margins.csv')
