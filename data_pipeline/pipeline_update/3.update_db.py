import psycopg2
import os

import datetime
from dateutil import tz
from datetime import timedelta

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

conn = psycopg2.connect(
    host = os.environ['host'],
    port = os.environ['port'],
    database = os.environ['db_name'],
    user = os.environ['user'],
    password = os.environ['db_password']
)

cursor = conn.cursor()
conn.autocommit = True

# Path to your CSV file
today_date = (datetime.datetime.now(tz=tz.gettz('Asia/Singapore'))).strftime('%Y%m%d')
csv_file_path = f'../../data/processed/{today_date}/price.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# Convert 'date' to datetime format
df['date'] = pd.to_datetime(df['date'])

# Convert 'price' to float format
df['price'] = df['price'].astype(float)

# Iterate over DataFrame rows using itertuples for better performance
for row in df.itertuples(index=False):
    cursor.execute("""
        INSERT INTO public.price (symbol, date, price)
        VALUES (%s, %s, %s)
        ON CONFLICT (symbol, date) DO NOTHING;
    """, ( row.symbol, row.date,row.price))

# update buy_price for portfolio table with EOD price from price table
update_query_1 = '''
UPDATE public.portfolio AS portfolio
SET buy_price = price_table.price
FROM public.price AS price_table
WHERE portfolio.symbol = price_table.symbol
AND portfolio.date = price_table.date
AND portfolio.buy_price IS NULL;
'''
cursor.execute(update_query_1)

# calculate units of bought for each symbol in a specific portfolio
update_query_2 = '''
UPDATE public.portfolio
SET units = amount_allocated / buy_price
WHERE buy_price IS NOT NULL;
'''

cursor.execute(update_query_2)

cursor.close()
