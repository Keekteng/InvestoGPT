import os
import json
import pandas as pd
import re

import datetime
from dateutil import tz

def process_raw_data():
    try:
        valid_symbols = pd.read_csv('../../data/processed/20240223/fund_detail.csv')['symbol'].unique()
        today_date = (datetime.datetime.now(tz=tz.gettz('Asia/Singapore'))).strftime('%Y%m%d')
        processed_dir = f'../../data/processed/{today_date}'

        os.makedirs(processed_dir, exist_ok=True)


        dir = f'../../data/raw/{today_date}/funds/'
        price_symbol_list = []
        date_list = []
        price_list = []


        for filename in os.listdir(dir):
            file_path = os.path.join(dir,filename)
            with open(file_path,'r') as f:
                fund_dict = json.load(f)
            pattern = r'\/([A-Z]+)\.json'
            match = re.search(pattern, file_path)
            symbol = match.group(1)

            if 'historical_price_table' in fund_dict:
                if 'date' in fund_dict['historical_price_table'] and 'price' in fund_dict['historical_price_table']:
                    for date,price in zip(fund_dict['historical_price_table']['date'],fund_dict['historical_price_table']['price']):
                        price_symbol_list.append(symbol)
                        date_list.append(date.strip())
                        price_list.append(float(price.strip().replace('$','')))
                
        price_df = pd.DataFrame({'symbol':price_symbol_list,'date':date_list,'price':price_list})

        price_df = price_df[price_df.symbol.isin(valid_symbols)].reset_index(drop=True)

        price_df['date'] = pd.to_datetime(price_df['date'])

        price_df.to_csv(f'../../data/processed/{today_date}/price.csv')
            
        print("Processing Script Completed!")

        return True
    except Exception as e:
        print(f"ERROR in Processing Script: {e}")
        return False



