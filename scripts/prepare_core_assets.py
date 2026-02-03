"""
prepare_core_assets.py

Pipeline script to prepare and save standardized core asset datasets
for use in the WealthPathAnalyzer.

- Reads, processes, and aligns core asset data (Gold, S&P500, CPI, BTC, Savings, Treasury)
- Calculates monthly returns and indices
- Aligns all DataFrames to a common valid date range for robust analysis

Author: Fatima Ezzahra Jabiri
Date created: 2026-01-30
"""

import sys
import os
import pandas as pd
from typing import Dict, Optional

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(PROJECT_ROOT)


# Path constants
PATHS = {
    'PROJECT_ROOT': PROJECT_ROOT,
    'DATA_DIR': os.path.join(PROJECT_ROOT, "data/processed/"),
    'DATA_RAW_DIR': os.path.join(PROJECT_ROOT, "data/raw/"),
    'SRC_DIR': os.path.join(PROJECT_ROOT, "src"),
    
    # Raw data files
    'SHILLER_DATA': os.path.join(PROJECT_ROOT, "data/raw/shiller_data.xls"),
    'SHILLER_RATES': os.path.join(PROJECT_ROOT, "data/raw/US_data_yearly.xlsx"),
    'GOLD_DATA': os.path.join(PROJECT_ROOT, "data/raw/gold_data_2025_monthly.csv"),
    'BTC_DATA': os.path.join(PROJECT_ROOT, "data/raw/btc-USD_monthly.csv"),
    
}

def validate_paths() -> None:
    """Validate that all required paths exist"""
    for path_name, path in PATHS.items():
        if not os.path.exists(path):
            raise ValueError(f"Required path not found - {path_name}: {path}")

from src.data_loader import *
from src.returns import *


# API key
api_key = '6eb6431c9f91a3dc605457629e5d85f0'



def main():

    # Validate all required paths before proceeding
    validate_paths()

    # === Step 1: Reading and Pre-processing data === # 

    # Read the btc price data: 
    print("1. Reading BTC price data...")
    btc_df = pd.read_csv(PATHS["BTC_DATA"], index_col= "Date")
    btc_df.index = pd.to_datetime(btc_df.index)
    btc_df["BTC-USD_Price"] = btc_df["Close"].copy()
    print(f"Date range: from {btc_df.index.min()} to {btc_df.index.max()}\n")
    print(f"Time series frequency: {btc_df.index.inferred_freq}")

    # Read and pre-process Shiller data:
    print("2. Loading Shiller data...")
    shiller_df = shiller_data(PATHS['SHILLER_DATA'])
    print(f"Date range: from {shiller_df.index.min()} to {shiller_df.index.max()}\n")
    print(f"Time series frequency: {shiller_df.index.inferred_freq}")

    # Read US annual risk free rates from Shiller data:

    print("3. Reading annual risk free rate data...")
    shiller_rates_df = shiller_annual_interest_rates(PATHS['SHILLER_RATES'])
    print(f"Date range: from {shiller_rates_df.index.min()} to {shiller_rates_df.index.max()}\n")
    print(f"Time series frequency: {shiller_rates_df.index.inferred_freq}")

    # Read and pre-process gold data:
    print("4. Reading gold price data ...")
    gold_df = process_gold_data(PATHS['GOLD_DATA'])
    print(f"Date range: from {gold_df.index.min()} to {gold_df.index.max()}\n")
    print(f"Time series frequency: {gold_df.index.inferred_freq}")
    
    # Fetch and rbuild the full cash monthly return series
    print("5. Fetching Fed Funds Rate & building monthly savings rate...")
    df_rates = fetch_fedfunds_series(api_key)
    savings_df = build_full_cash_return_series(shiller_df, shiller_rates_df, df_rates)
    print(f"Date range: from {savings_df.index.min()} to {savings_df.index.max()}\n")
    print(f"Time series frequency: {savings_df.index.inferred_freq}")
    print("Few rows:\n")
    print(savings_df.head(3))

    # === Step 2: Extracting 10Y bond data, sp500 data and CPI data === #

    sp500_df = extract_asset_df(shiller_df, "sp500")
    print("First rows:\n")
    print(sp500_df.head(3))

    bonds_df = extract_asset_df(shiller_df, "10Y bond")
    print("First rows:\n")
    print(bonds_df.head(3))

    cpi_df = load_cpi_from_shiller(shiller_df)
    print("First rows:\n")
    print(cpi_df.head(3))

    # === Step 3: Calculating returns for all core assets === #
    # 3.1 S&P500 returns
    sp500_df = compute_equity_returns(sp500_df, price_col="sp500_Price", dividend_col="sp500_Dividend", asset_name="sp500")
    # 3.2 10Y bond returns
    bonds_df = compute_bond_returns(bonds_df, yield_col="10Y_Treasury_Annual_Rate (%)", asset_name="10Y bond")
    # 3.3 Gold returns   
    gold_df = compute_price_returns(gold_df, price_col="Gold_Price", asset_name="Gold")
    # 3.4 BTC returns 
    btc_df = compute_price_returns(btc_df, price_col="BTC-USD_Price", asset_name="BTC-USD")

    # === Step 4: Calculating Total Return Index and Price Return Index for all assets === #
    # 4.1 S&P500 Total Return Index and Price Return Index
    sp500_df = build_price_index(sp500_df, return_col="sp500_Price_Return", asset_name="sp500")
    sp500_df = build_total_return_index(sp500_df, return_col="sp500_Total_Return", asset_name="sp500")
    # 4.2 10Y bond Total Return Index and Price Return Index
    bonds_df = build_price_index(bonds_df, return_col="10Y bond_Price_Return", asset_name="10Y bond")
    bonds_df = build_total_return_index(bonds_df, return_col="10Y bond_Total_Return", asset_name="10Y bond")
    # 4.3 Gold Price Return Index
    gold_df = build_price_index(gold_df, return_col="Gold_Return", asset_name="Gold")
    # 4.4 BTC Price Return Index
    btc_df = build_price_index(btc_df, return_col="BTC-USD_Return", asset_name="BTC-USD")
    # 4.5 Savings Total Return Index
    savings_df = build_total_return_index(savings_df, return_col="Cash_Monthly_Rate (decimal)", asset_name="Cash")
    
    # === Step 5: Align all DataFrames to common valid date range and save outputs === #
    print("5. Aligning all DataFrames to common valid date range and saving outputs (Except BTC-USD as it has different date range)...")

    start_date = max(
    sp500_df.index.min(),
    bonds_df.index.min(),
    gold_df.index.min(),
    savings_df.index.min(),
    cpi_df.index.min()
    )
    end_date = pd.Timestamp("2025-12-31")

    sp500_df = sp500_df.loc[start_date:end_date]
    bonds_df = bonds_df.loc[start_date:end_date]
    gold_df = gold_df.loc[start_date:end_date]
    savings_df = savings_df.loc[start_date:end_date]
    cpi_df = cpi_df.loc[start_date:end_date]
    btc_df = btc_df.loc[btc_df.index.min():end_date]
    
    # Print few last rows of each asset index returns. For btc print first and last 3 rows
    #print("S&P500 returns:\n", sp500_df[[ "sp500_Price", "sp500_Total_Return", "sp500_Price_Return", "sp500_TR_Index", "sp500_Price_Index"]].tail(3))
    #print("10Y bond returns:\n", bonds_df[[ "10Y bond_Price", "10Y bond_Total_Return", "10Y bond_Price_Return", "10Y bond_TR_Index", "10Y bond_Price_Index"]].tail(3))
    #print("Gold returns:\n", gold_df[[ "Gold_Price", "Gold_Return", "Gold_Price_Index"]].tail(3))
    #print("BTC returns:\n", btc_df[[ "BTC-USD_Price", "BTC-USD_Return", "BTC-USD_Price_Index"]].head(3))
    #print("BTC returns:\n", btc_df[[ "BTC-USD_Price", "BTC-USD_Return", "BTC-USD_Price_Index"]].tail(3))
    #print("Savings returns:\n", savings_df[[ "Cash_Monthly_Rate (decimal)", "Cash_TR_Index"]].tail(3))

    # Concatenate all core asset DataFrames into a single DataFrame for saving
    master_df = pd.concat([sp500_df, bonds_df, gold_df, btc_df, savings_df, cpi_df], axis=1)

    # === Step 6: Save all processed core asset DataFrames to CSV files === #
    output_dir = PATHS['DATA_DIR']
    os.makedirs(output_dir, exist_ok=True)
    master_df.to_csv(os.path.join(output_dir, "core_assets_master.csv"))

    metadata = create_metadata(master_df)
    save_metadata(metadata, os.path.join(output_dir, "core_assets_master_metadata.json"))

if __name__ == "__main__":
    
    main()
