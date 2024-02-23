import os
import json
import pandas as pd
import re

import datetime
from dateutil import tz

# def perc_to_decimal(string):
#     if string:
#         return round(float((string.strip().replace('%','')))/100,5)
#     else:
#         return string
    
# def clean_risk_level(string):
#     risk_level = int(string.strip())
#     if risk_level == 1:
#         return 'Conservative'
#     elif risk_level == 2:
#         return 'Conservative to Moderate'
#     elif risk_level == 3:
#         return 'Moderate'
#     elif risk_level == 4:
#         return 'Moderate to Aggressive'
#     else:
#         return 'Aggressive'
    
# def clean_min_investment(string):
#     return int(float(string.replace(',','').replace('$','')))


valid_symbols = pd.read_csv('../../data/processed/20240223/fund_detail.csv')['symbol'].tolist()
today_date = (datetime.datetime.now(tz=tz.gettz('Asia/Singapore'))).strftime('%Y%m%d')
processed_dir = f'../../data/processed/{today_date}'

os.makedirs(processed_dir, exist_ok=True)

# dir = f'../../data/raw/{today_date}/funds/'

# symbol_list = []
# category = []
# asset_class = []
# inception_date = []
# min_inv = []
# exp_ratio = []
# return_ytd = []
# return_1_year = []
# return_3_year = []
# return_5_year = []
# return_10_year = []
# return_inception = []
# return_bench_ytd = []
# return_bench_1_year = []
# return_bench_3_year = []
# return_bench_5_year = []
# return_bench_10_year = []
# return_bench_inception = []
# risk_level = []
# product_summary = []
# date = []
# price = []

# for filename in os.listdir(dir):
#     file_path = os.path.join(dir,filename)
#     with open(file_path,'r') as f:
#         fund_dict = json.load(f)
#     pattern = r'\/([A-Z]+)\.json'
#     match = re.search(pattern, file_path)
#     symbol = match.group(1)
#     symbol_list.append(symbol)
    
#     category.append(fund_dict['key_fact_table']['Category'])
#     asset_class.append(fund_dict['key_fact_table']['Asset class'])

#     if 'product_summary' in fund_dict:
#         product_summary.append(fund_dict['product_summary'])
#     else:
#         product_summary.append("NA")

#     if 'Inception date' in fund_dict['key_fact_table']:
#         inception_date.append(fund_dict['key_fact_table']['Inception date'])
#     else:
#         inception_date.append(None)

#     if 'risk_level' in fund_dict:
#         risk_level.append(fund_dict['risk_level'])
#     else:
#         risk_level.append(None)

#     if 'perf_table' in fund_dict:
#         perf_table = fund_dict['perf_table']

#         if symbol in perf_table:
#             symbol_key = symbol
#         elif symbol+'1' in perf_table:
#             symbol_key = symbol + '1'
#         elif symbol+'2' in perf_table:
#             symbol_key = symbol+'2'
#         elif symbol+'4' in perf_table:
#             symbol_key = symbol+'4'
#         else:
#             symbol_key = symbol+' (Market price)'
        
#         if 'Benchmark1' in perf_table:
#             benchmark_key = 'Benchmark1'
#         elif 'Benchmark3' in perf_table:
#             benchmark_key = 'Benchmark3'
#         else:
#             benchmark_key = 'Benchmark'

#         for row in zip(perf_table['index'],perf_table[symbol_key],perf_table[benchmark_key]):
#             if row[0]=="YTD":
#                 return_ytd.append(row[1])
#                 return_bench_ytd.append(row[2])
#             if row[0]=="1-yr":
#                 return_1_year.append(row[1])
#                 return_bench_1_year.append(row[2])
#             if row[0]=="3-yr":
#                 return_3_year.append(row[1])
#                 return_bench_3_year.append(row[2])
#             if row[0]=="5-yr":
#                 return_5_year.append(row[1])
#                 return_bench_5_year.append(row[2])
#             if row[0]=="10-yr":
#                 return_10_year.append(row[1])
#                 return_bench_10_year.append(row[2])
#             if row[0]=="Since inception":
#                 return_inception.append(row[1]) 
#                 return_bench_inception.append(row[2])
#     else:
#         return_ytd.append(None)
#         return_bench_ytd.append(None)
#         return_1_year.append(None)
#         return_bench_1_year.append(None)
#         return_3_year.append(None)
#         return_bench_3_year.append(None)
#         return_5_year.append(None)
#         return_bench_5_year.append(None)
#         return_10_year.append(None)
#         return_bench_10_year.append(None)
#         return_inception.append(None)
#         return_bench_inception.append(None)

#     if 'min_investment' in fund_dict:
#         min_inv.append(fund_dict['min_investment'])
#     else:
#         min_inv.append(None)
    
#     if 'exp_ratio' in fund_dict:
#         exp_ratio.append(fund_dict['exp_ratio'])
#     else:
#         exp_ratio.append(None)

#     if 'date' in fund_dict['historical_price_table'] and 'price' in fund_dict['historical_price_table']:
#         date.append(fund_dict['historical_price_table']['date'])
#         price.append(fund_dict['historical_price_table']['price'])
#     else:
#         date.append(None)
#         price.append(None)


# table_col = ['symbol','category','product_summary','asset_class','inception_date','minimum_investment','expense_ratio',
#              'fund_return_ytd','average_annual_fund_return_for_1_year','average_annual_fund_return_for_3_year',
#              'average_annual_fund_return_for_5_year','average_annual_fund_return_for_10_year','average_annual_fund_return_since_inception',
#              'benchmark_return_ytd','average_annual_benchmark_return_for_1_year','average_annual_benchmark_return_for_3_year','average_annual_benchmark_return_for_5_year',
#              'average_annual_benchmark_return_for_10_year','average_annual_benchmark_return_since_inception','risk_level','date','price']

# fund_df = pd.DataFrame(columns=table_col,data=list(zip(symbol_list,category,product_summary,asset_class,inception_date,min_inv,
#                                                        exp_ratio,return_ytd,return_1_year,return_3_year,return_5_year,return_10_year,
#                                                        return_inception,return_bench_ytd,return_bench_1_year,return_bench_3_year,
#                                                        return_bench_5_year,return_bench_10_year,return_bench_inception,risk_level,date,price)))

# # Drop all funds with missing information
# fund_df.dropna(subset=['date','price'],inplace=True)

# # Remove funds with null min_inv --> Closed not allowed to be bought by investor anymore
# cleaned_df = fund_df.copy()
# cleaned_df = cleaned_df[cleaned_df['minimum_investment'].notna()]

# # replace unicode dash with Null value
# cleaned_df = cleaned_df.replace('\u2014',None)

# # clean risk level
# cleaned_df['risk_level'] = cleaned_df['risk_level'].apply(clean_risk_level)

# # convert percentages string to float
# percentage_cols = ['expense_ratio','fund_return_ytd','average_annual_fund_return_for_1_year','average_annual_fund_return_for_3_year','average_annual_fund_return_for_5_year','average_annual_fund_return_for_10_year','average_annual_fund_return_since_inception','benchmark_return_ytd','average_annual_benchmark_return_for_1_year','average_annual_benchmark_return_for_3_year','average_annual_benchmark_return_for_5_year','average_annual_benchmark_return_for_10_year','average_annual_benchmark_return_since_inception']
# for col in percentage_cols:
#     cleaned_df[col] = cleaned_df[col].apply(perc_to_decimal)

# cleaned_df['inception_date'] = pd.to_datetime(cleaned_df['inception_date'])

# cleaned_df['minimum_investment'] = cleaned_df['minimum_investment'].apply(clean_min_investment)


# perf_df = cleaned_df[['symbol','fund_return_ytd','average_annual_fund_return_for_1_year','average_annual_fund_return_for_3_year',
#                       'average_annual_fund_return_for_5_year','average_annual_fund_return_for_10_year','average_annual_fund_return_since_inception',
#                       'benchmark_return_ytd','average_annual_benchmark_return_for_1_year','average_annual_benchmark_return_for_3_year','average_annual_benchmark_return_for_5_year',
#                       'average_annual_benchmark_return_for_10_year','average_annual_benchmark_return_since_inception']].reset_index(drop=True).copy()

# fund_detail_df = cleaned_df[['symbol','category','product_summary','asset_class','inception_date','minimum_investment','expense_ratio','risk_level']].reset_index(drop=True).copy()
# perf_df.to_csv(f'../../data/processed/{today_date}/perf.csv')
# fund_detail_df.to_csv(f'../../data/processed/{today_date}/fund_detail.csv')

dir = f'../../data/raw/{today_date}/funds/'
# sec_symbol_list = []
# sec_list = []
# sec_alloc = []
# reg_symbol_list = []
# reg_list = []
# reg_alloc = []
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
    # symbol_list.append(symbol)
    
    # if 'weighted_exposure_table' in fund_dict:
    #     if ' Sectors' in fund_dict['weighted_exposure_table']:
    #         for sec,alloc in zip(fund_dict['weighted_exposure_table'][' Sectors'][' Sectors'],fund_dict['weighted_exposure_table'][' Sectors'][symbol]):
    #             sec_symbol_list.append(symbol)
    #             sec_list.append(sec.strip())
    #             sec_alloc.append(alloc.strip())
                
    #     if ' Regions' in fund_dict['weighted_exposure_table']:
    #         for reg,alloc in zip(fund_dict['weighted_exposure_table'][' Regions'][' Regions'],fund_dict['weighted_exposure_table'][' Regions'][symbol]):
    #             reg_symbol_list.append(symbol)
    #             reg_list.append(reg.strip())
    #             reg_alloc.append(alloc.strip())
    #     else:
    #         reg_symbol_list.append(symbol)
    #         reg_list.append('North America')
    #         reg_alloc.append("100%")

    if 'historical_price_table' in fund_dict:
        if 'date' in fund_dict['historical_price_table'] and 'price' in fund_dict['historical_price_table']:
            for date,price in zip(fund_dict['historical_price_table']['date'],fund_dict['historical_price_table']['price']):
                price_symbol_list.append(symbol)
                date_list.append(date.strip())
                price_list.append(float(price.strip().replace('$','')))
        
# sector_df = pd.DataFrame({'symbol':sec_symbol_list,'sector':sec_list,'allocation':sec_alloc})
# region_df = pd.DataFrame({'symbol':reg_symbol_list,'region':reg_list,'allocation':reg_alloc})
price_df = pd.DataFrame({'symbol':price_symbol_list,'date':date_list,'price':price_list})

# valid_symbols = cleaned_df['symbol'].tolist()
# sector_df = sector_df[sector_df['symbol'].isin(valid_symbols)].reset_index(drop=True)
# region_df = region_df[region_df['symbol'].isin(valid_symbols)].reset_index(drop=True)
price_df = price_df[price_df.symbol.isin(valid_symbols)].reset_index(drop=True)

# sector_df['allocation'] = sector_df['allocation'].apply(perc_to_decimal)
# region_df['allocation'] = region_df['allocation'].apply(perc_to_decimal)
price_df['date'] = pd.to_datetime(price_df['date'])

# sector_df.to_csv(f'../../data/processed/{today_date}/sector.csv')
# region_df.to_csv(f'../../data/processed/{today_date}/region.csv')
price_df.to_csv(f'../../data/processed/{today_date}/price.csv')
    




