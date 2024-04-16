import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append('../../')

from data_pipeline.pipeline_update.pipeline_1_scrape_latest_price import scrape_raw_data
from data_pipeline.pipeline_update.pipeline_2_processing import process_raw_data
from data_pipeline.pipeline_update.pipeline_3_update_db import update_database


def overall_pipeline():

    # Step 1: Scrape latest NAV for all funds
    if scrape_raw_data():
        # Step 2: Clean and Process Raw Data
        if process_raw_data():
            # Step 3: Update price table and portfolio table with latest NAV
            update_database()

if __name__=='__main__':
    overall_pipeline()