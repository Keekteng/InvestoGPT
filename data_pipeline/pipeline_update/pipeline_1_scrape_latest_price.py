from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium_stealth import stealth

import datetime
from dateutil import tz
import time
import random
import json
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


import undetected_chromedriver as uc

processed_dir_path = '../../data/processed'

directories = [d for d in os.listdir(processed_dir_path) if os.path.isdir(os.path.join(processed_dir_path, d))]
latest_date = max(datetime.datetime.strptime(d, '%Y%m%d') for d in directories)

# Format the datetime object back to a string in the original format
latest_date_str = latest_date.strftime('%Y%m%d')

today_date = (datetime.datetime.now(tz=tz.gettz('Asia/Singapore'))).strftime('%Y%m%d')
urls_output_dir = f'../../data/raw/{today_date}/urls'
funds_output_dir = f'../../data/raw/{today_date}/funds'

os.makedirs(urls_output_dir, exist_ok=True)
os.makedirs(funds_output_dir, exist_ok=True)

# selenium stealth driver used for scraping 
def create_driver(debug=False):

    options = webdriver.ChromeOptions()
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--start-maximized")
    if debug==False:
        options.add_argument("--headless")

    driver = uc.Chrome(
        options=options
    )
    stealth(driver,
            # user_agent=agent,
            languages=["en-US", "en"],
            vendor="Google Inc.",
            platform="Win32",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True,
            )
    return driver

def scrape_latest_price(html_page_source,fund_detail,driver):

    # Only fill in missing price for symbols that have price table
    if 'date' in fund_detail['historical_price_table'] and 'price' in fund_detail['historical_price_table']:

        if html_page_source.find(name='div',attrs={'id':'mat-tab-label-0-0'}):
            tab_1_month = driver.find_element(By.ID,'mat-tab-label-0-0')    
            tab_1_month.location_once_scrolled_into_view
            time.sleep(random.uniform(1,2))
            tab_1_month.click()
            time.sleep(random.uniform(2,3))

            radio_button = driver.find_element(By.XPATH,"//input[@type='radio' and @aria-label='table']")
            radio_button.click()
            time.sleep(random.uniform(2,4))

            listbox_location = driver.find_element(By.XPATH,"//button[@tabindex='0' and @aria-haspopup='listbox']")
            listbox_location.click()
            time.sleep(random.uniform(0,0.5))

            overlay_container = driver.find_element(By.XPATH,"//div[@class='cdk-overlay-container']")
            driver.find_element(By.XPATH,"//table[@aria-label='Historical prices table']").location_once_scrolled_into_view
            time.sleep(random.uniform(1,2))
            listbox = overlay_container.find_element(By.XPATH,"//div[@role='listbox' and @tabindex='0']")
            num_options = len(listbox.find_elements(By.TAG_NAME,'vui-option'))

            # Go to the last page instantly as we only want latest price
            option = listbox.find_elements(By.TAG_NAME,'vui-option')[-1]
            option.location_once_scrolled_into_view
            time.sleep(random.uniform(0.5,1))
            option.click()
            time.sleep(random.uniform(0.5,1))
            updated_page_source = BeautifulSoup(driver.page_source)
            price = updated_page_source.find(name='div',attrs={'id':'price_section'})
            for tr in price.find(name='table',attrs={'aria-label':'Historical prices table'}).find('tbody').find_all('tr'):
                cur_date = tr.find(name='td',attrs={'data-rpa-tag-id':'historicalDate'}).text
                cur_price = tr.find(name='td',attrs={'data-rpa-tag-id':'historicalPrice'}).text

                # logic to update missing dates and corresponding prices
                if cur_date not in fund_detail['historical_price_table']['date']:
                    fund_detail['historical_price_table']['date'].append(cur_date)
                    fund_detail['historical_price_table']['price'].append(cur_price)
                
    return fund_detail

def scrape_raw_data():

    with open(f'../../data/raw/{latest_date_str}/urls/funds_url.json','r')as f:
        funds_url = json.load(f)

    try:
        for index,(symbol,url) in enumerate(tqdm(funds_url.items(),position=0,leave=True)):
            # if index<89:
            #     continue
            with open(f'../../data/raw/{latest_date_str}/funds/{symbol}.json','r')as f:
                fund_detail = json.load(f)
            driver = create_driver()
            driver.get(url)
            time.sleep(random.uniform(1,1.5))
            soup = BeautifulSoup(driver.page_source)
            fund_detail = scrape_latest_price(soup,fund_detail,driver)
            with open(f"{funds_output_dir}/{symbol}.json",'w') as f:
                json.dump(fund_detail,f)
            driver.quit()
        
        with open(f"{urls_output_dir}/funds_url.json",'w')as f:
            json.dump(funds_url,f)
        
        return True
            
    except Exception as e:
        print(f"ERROR --> {symbol}, Index --> {index}")
        print(e)
        return False
