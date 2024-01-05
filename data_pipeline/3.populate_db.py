import psycopg2
import os


from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from sqlalchemy import create_engine,Date

engine = create_engine(
    f"postgresql+psycopg2://{os.environ['user']}:{os.environ['db_password']}@{os.environ['host']}:{os.environ['port']}/{os.environ['db_name']}",
    )

perf_df = pd.read_csv('../data/processed/20231212/perf.csv').drop(columns=['Unnamed: 0'])
fund_detail_df = pd.read_csv('../data/processed/20231212/fund_detail.csv').drop(columns=['Unnamed: 0'])
region_df = pd.read_csv('../data/processed/20231212/region.csv').drop(columns=['Unnamed: 0'])
sector_df = pd.read_csv('../data/processed/20231212/sector.csv').drop(columns=['Unnamed: 0'])

dtype = {
    'inception_date': Date
}

perf_df.to_sql('performance', engine, if_exists='replace')
fund_detail_df.to_sql('fund_detail', engine, if_exists='replace',dtype=dtype)
region_df.to_sql('region',engine, if_exists='replace')
sector_df.to_sql('sector',engine,if_exists='replace')


