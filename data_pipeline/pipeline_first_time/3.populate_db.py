import psycopg2
import os


from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from sqlalchemy import create_engine,Date

engine = create_engine(
    f"postgresql+psycopg2://{os.environ['user']}:{os.environ['db_password']}@{os.environ['host']}:{os.environ['port']}/{os.environ['db_name']}",
    )

conn = psycopg2.connect(
    host = os.environ['host'],
    port = os.environ['port'],
    database = os.environ['db_name'],
    user = os.environ['user'],
    password = os.environ['db_password']
)

cursor = conn.cursor()

perf_df = pd.read_csv('../../data/processed/20231212/perf.csv').drop(columns=['Unnamed: 0'])
fund_detail_df = pd.read_csv('../../data/processed/20231212/fund_detail.csv').drop(columns=['Unnamed: 0'])
region_df = pd.read_csv('../../data/processed/20231212/region.csv').drop(columns=['Unnamed: 0'])
sector_df = pd.read_csv('../../data/processed/20231212/sector.csv').drop(columns=['Unnamed: 0'])
price_df = pd.read_csv('../../data/processed/20231212/price.csv').drop(columns=['Unnamed: 0'])

dtype_fund_detail = {
    'inception_date': Date
}

dtype_price = {
    'date': Date
}

perf_df.to_sql('performance', engine, if_exists='replace')
fund_detail_df.to_sql('fund_detail', engine, if_exists='replace',dtype=dtype_fund_detail)
region_df.to_sql('region',engine, if_exists='replace')
sector_df.to_sql('sector',engine,if_exists='replace')
price_df.to_sql('price',engine,if_exists='replace',dtype=dtype_price)

create_portfolio_table_query = '''
CREATE TABLE IF NOT EXISTS public.portfolio (
    Date DATE NOT NULL,
    portfolio_id int8 NOT NULL,
    username varchar NOT NULL,
    symbol varchar NOT NULL,
    buy_price float8 NULL,
    units float8 NULL,
    amount_allocated float8 NOT NULL,
    CONSTRAINT portfolio_pkey PRIMARY KEY (portfolio_id, username, symbol)
    );
'''

cursor.execute(create_portfolio_table_query)
conn.commit()
cursor.close()


