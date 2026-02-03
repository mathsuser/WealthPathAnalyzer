"""
data_loader.py

Core data loading and preprocessing functions for WealthPathAnalyzer.

- Load and standardize financial data (Shiller, FRED, Yahoo Finance, gold, BTC, CPI, cash, bonds)
- Align and merge asset DataFrames for unified analysis
- Convert annual to monthly rates
- Create and save minimal metadata for datasets

Functions:
    fetch_yahoo_data, shiller_data, shiller_annual_interest_rates, fetch_fedfunds_series,
    build_full_cash_return_series, process_gold_data, load_cpi_from_shiller, extract_asset_df,
    convert_annual_to_monthly_rate, create_metadata, save_metadata

Author: Fatima Ezzahra Jabiri
Created: 2026-02-01

"""

import pandas as pd
from fredapi import Fred
import numpy as np
import yfinance as yf
import json
from datetime import datetime

def fetch_yahoo_data(
    ticker,
    start_date,
    end_date=None,
    freq="MS",
    auto_adjust=False,
    include_dividends=True
):
    """
    Fetches monthly price and (optionally) dividend data for a Yahoo Finance ticker.

    Parameters:
        ticker (str): Yahoo Finance ticker symbol.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str or None): End date (YYYY-MM-DD), default today.
        freq (str): 'MS' for month start, 'M' for month end.
        auto_adjust (bool): If True, 'Close' is adjusted (total return); else raw close.
        include_dividends (bool): If True, include monthly dividends.

    Returns:
        pd.DataFrame: DataFrame with columns ['Price', 'Dividend'] (if available), indexed by month.
    """
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=auto_adjust,
        progress=False,
        actions=include_dividends
    )

    # Flatten MultiIndex columns (if any)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [' '.join([str(i) for i in col if i]).strip() for col in data.columns.values]

    # Build column names (handles both single and multi-ticker cases)
    price_col = f"Close {ticker}" if f"Close {ticker}" in data.columns else "Close"
    dividend_col = f"Dividends {ticker}" if f"Dividends {ticker}" in data.columns else "Dividends"

    if price_col not in data.columns:
        raise ValueError(f"{price_col} not found in Yahoo data for {ticker}")

    # Resample price
    if freq == "MS":
        price = data[price_col].resample(freq).first()
    else:
        price = data[price_col].resample(freq).last()

    # Dividends (if requested and available)
    if include_dividends and dividend_col in data.columns:
        dividends = data[dividend_col].resample(freq).sum()
        df = pd.DataFrame({"Price": price, "Dividend": dividends})
    else:
        df = pd.DataFrame({"Price": price})

    df = df.loc[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df.loc[df.index <= pd.to_datetime(end_date)]

    return df.dropna(subset=["Price"], how="all")

def pivot_asset_long_to_wide(df):
    # Assumes columns: ['Ticker', 'Data_Type', 'Value']
    wide = df.pivot_table(
        index=df.index,
        columns='Data_Type',
        values='Value',
        aggfunc='first'
    )
    # Rename columns to match expected: 'Price', 'Adj Close', 'Dividend'
    wide = wide.rename(columns={
        'Close': 'Price',
        'Adj_Close': 'Adj Close',
        'Dividends': 'Dividend'
    })
    return wide

# Load Shiller data from an Excel file and process it
def shiller_data(filename):
    """
    Load Shiller data from the specified Excel file.
    Args:
        filename (str): Path to the Excel file containing Shiller data.
    Returns:
        pd.DataFrame: DataFrame containing the relevant Shiller data.
    """
    xls = pd.ExcelFile(filename)
    df = pd.read_excel(xls, sheet_name=xls.sheet_names[1], skiprows=7)
    relevant_cols = [
        'Date', 'P', 'D', 'E', 'CPI', 'Fraction', 'Rate GS10',
        'Price', 'Dividend', 'Price.1', 'Earnings', 'Earnings.1']
    df = df[relevant_cols]
    df = df.rename(columns={
        'Date': 'DateStr',
        'Fraction': 'DateDecimal',
        'Rate GS10': '10Y_Treasury_Rate',
        'Price': 'Real P',
        'Dividend': 'Real D',
        'Price.1': 'Real TR P',
        'Earnings': 'Real E',
        'Earnings.1': 'Real TR Scaled E'})
    data = df.dropna(subset=["DateStr", "DateDecimal"]).copy()
    data["Year"] = data["DateDecimal"].astype(int)
    data["Month"] = ((data["DateDecimal"] - data["Year"]) * 12 + 1).astype(int)
    data["Date"] = pd.to_datetime(dict(year=data["Year"], month=data["Month"], day=1)) + pd.offsets.MonthEnd(0)
    data.set_index("Date", inplace=True)
    data.drop(columns=["Year", "Month", "DateStr", "DateDecimal"], inplace=True)
    # Convert relevant columns to numeric
    data = data.apply(pd.to_numeric, errors="coerce")
    return data

# Load Shiller annual interest rates from an Excel file
def shiller_annual_interest_rates(filename):
    """
    Load Shiller annual interest rates from the specified Excel file.
    Args:
        filename (str): Path to the Excel file containing Shiller annual interest rates.
    Returns:
        pd.DataFrame: DataFrame containing the relevant annual interest rates.
    """
    xls = pd.ExcelFile(filename)
    df = pd.read_excel(xls, sheet_name=xls.sheet_names[1], skiprows=7)
    df = df[['Unnamed: 0', 'Unnamed: 4', 'Unnamed: 7']]
    df = df.rename(columns={
        'Unnamed: 0': 'Year',
        'Unnamed: 4': '1Y_Nominal_Rate',
        'Unnamed: 7': '1Y_Real_Rate'})
    df = df.dropna(subset=["Year"]).copy()
    df["Date"] = pd.to_datetime(df["Year"].astype(int), format='%Y') + pd.offsets.YearEnd(0)
    df.set_index("Date", inplace=True)
    df.drop(columns=["Year"], inplace=True)
    return df


def fetch_fedfunds_series(api_key: str, start_date: str = None, end_date: str = None, period: str = "monthly") -> pd.DataFrame:
    """
    Fetch the Federal Funds Rate series from FRED and calculate the specified rate.
    Args:
        api_key (str): FRED API key.
        start_date (str): Start date for the data in 'YYYY-MM-DD' format.
        end_date (str): End date for the data in 'YYYY-MM-DD' format.
        period (str): The period for the rate ('daily', 'monthly', 'quarterly', 'annually').
    Returns:
        pd.DataFrame: DataFrame containing the Federal Funds Rate and the specified rate.
    """
    period_map = {'daily': 365, 'monthly': 12, 'quarterly': 4, 'annually': 1}
    if period not in period_map:
        raise ValueError("Period must be one of: 'daily', 'monthly', 'quarterly', 'annually'")
    periods_per_year = period_map[period]
    fred = Fred(api_key=api_key)
    data = fred.get_series('FEDFUNDS', observation_start=start_date, observation_end=end_date)
    df = data.to_frame(name='FEDFUNDS')
    df[f'{period}_rate'] = (1 + df['FEDFUNDS'] / 100) ** (1 / periods_per_year) - 1
    return df


def build_full_cash_return_series(df_shiller, df_annual_rates, df_fedfunds, spread=0.0, default_1870_rate=6.35):
    """
    Build a full cash return series using Shiller data, annual interest rates, and Federal Funds data.
    Args:
        df_shiller (pd.DataFrame): Shiller data DataFrame.
        df_annual_rates (pd.DataFrame): Annual interest rates DataFrame.
        df_fedfunds (pd.DataFrame): Federal Funds data DataFrame.
        spread (float): Spread to apply to the rates.
        default_1870_rate (float): Default rate for 1870.
    Returns:
        pd.DataFrame: DataFrame containing the full cash return series.
    """
    df_annual = df_annual_rates.copy()
    first_row = pd.DataFrame({"1Y_Nominal_Rate": [default_1870_rate]}, index=[pd.to_datetime('1870-01-31')])
    df_annual = df_annual[df_annual.index <= '2011-12-31']
    df_annual = pd.concat([first_row, df_annual])
    df_annual["Cash_Annual_Rate"] = (df_annual["1Y_Nominal_Rate"] - spread).clip(lower=0.0)
    # FIX: Resample to month-end instead of month-start
    df_monthly = df_annual.resample('ME').ffill()
    df_monthly["Cash_Monthly_Rate"] = (1 + df_monthly["Cash_Annual_Rate"] / 100) ** (1/12) - 1

    df_fedfunds_post2011 = df_fedfunds[df_fedfunds.index >= '2011-12-31'].copy()
    df_fedfunds_post2011["Cash_Annual_Rate"] = (df_fedfunds_post2011["FEDFUNDS"] - spread).clip(lower=0.0)
    df_fedfunds_post2011["Cash_Monthly_Rate"] = (df_fedfunds_post2011["monthly_rate"] - spread).clip(lower=0.0)

    df_combined = pd.concat([
        df_monthly[["Cash_Annual_Rate", "Cash_Monthly_Rate"]],
        df_fedfunds_post2011[["Cash_Annual_Rate", "Cash_Monthly_Rate"]]
    ])
    df_combined = df_combined.sort_index()
    df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
    df_combined = df_combined.reindex(df_shiller.index, method='ffill')

    # Final renaming and adding decimal monthly rate
    df_combined["Cash_Monthly_Rate (%)"] = df_combined["Cash_Monthly_Rate"] * 100
    df_combined.rename(columns={"Cash_Annual_Rate": "Cash_Annual_Rate (%)",
                                "Cash_Monthly_Rate": "Cash_Monthly_Rate (decimal)"}, inplace=True)
    return df_combined

# Load gold data from an Excel file and process it
def process_gold_data(filename):
    """
    Processes the gold data from a CSV or Excel file and filters it to start from 1871-01-01.
    Supports both .csv and .xls/.xlsx formats.
    Parameters:
        filename (str): Path to the gold data file (CSV or Excel).
    Returns:
        pd.DataFrame: Filtered DataFrame with gold prices from 1871 onwards.
    """
    import pandas as pd
    # Detect file type
    if filename.lower().endswith('.csv'):
        df = pd.read_csv(filename)
    else:
        xls = pd.ExcelFile(filename)
        df = pd.read_excel(xls, skiprows=1)
    # Rename columns for clarity
    df.columns = ["Date", "Gold_Price"]
    # Convert 'Date' to datetime and set to end-of-month
    df["Date"] = pd.to_datetime(df["Date"]) + pd.offsets.MonthEnd(0)
    df.set_index("Date", inplace=True)
    # Filter to only include data from 1871-01-01 onwards
    #df = df[df.index >= "1871-01-01"]
    return df


def load_cpi_from_shiller(shiller_df, cpi_col="CPI"):
    """
    Extracts and cleans CPI from Shiller data.

    Parameters:
        shiller_df (pd.DataFrame): Full Shiller data.
        cpi_col (str): Name of the CPI column.

    Returns:
        pd.DataFrame: Clean CPI series with datetime index.
    """
    try: 
        cpi = shiller_df[[cpi_col]].copy()
        cpi[cpi_col] = pd.to_numeric(cpi[cpi_col], errors="coerce")
        cpi[cpi_col] = cpi[cpi_col].interpolate(method="linear", limit_direction="both").ffill()
        return cpi
    except KeyError:
        print(f"Warning: Column {cpi_col} not found in Shiller data. Return empty DataFrame.")
        return pd.DataFrame()


def extract_asset_df(shiller_data, asset_type, start_date=None):
    """
    Extracts the sp500 price levels and dividends, and 10Y treasury yields from shiller data. 

    """
    
    if asset_type == "sp500":
        df = shiller_data[["P", "D"]].copy()
        df.rename(columns = {"P": "sp500_Price", "D": "sp500_Dividend"}, inplace = True)


    elif asset_type =="10Y bond":
        df = shiller_data[["10Y_Treasury_Rate"]].copy()
        df = convert_annual_to_monthly_rate(df, "10Y_Treasury_Rate", out_col= "10Y_Monthly_Rate", as_percent=False)
        df.rename(columns = {"10Y_Monthly_Rate": "10Y_Monthly_Rate (decimal)",
                             "10Y_Treasury_Rate": "10Y_Treasury_Annual_Rate (%)"}, inplace=True)
        df["10Y_Monthly_Rate (%)"] = df["10Y_Monthly_Rate (decimal)"]*100
        

    else: 
        raise ValueError("asset_type must be 'sp500' or '10Y bond'")

    if start_date:
        mask =df.index >= pd.to_datetime(start_date)
        df = df.loc[mask]

    return df            


def convert_annual_to_monthly_rate(df, annual_col, out_col="Monthly_Rate", method="linear", as_percent=False):
    """
    Converts annual % rate to monthly rate.

    Parameters:
        df (pd.DataFrame): DataFrame with annual rate in percent (e.g., 5.2 for 5.2%).
        annual_col (str): Column name with annual rate.
        out_col (str): Output column name for monthly rate.
        method (str): 'compound' for exact, 'linear' for Shiller-style (annual / 12 / 100).
        as_percent (bool): If True, return in percent instead of decimal.

    Returns:
        pd.DataFrame: DataFrame with added monthly rate column.
    """
    df = df.copy()
    if method == "compound":
        monthly = (1 + df[annual_col] / 100) ** (1/12) - 1
    elif method == "linear":
        monthly = df[annual_col] / 1200
    else:
        raise ValueError("method must be 'compound' or 'linear'")

    df[out_col] = monthly * 100 if as_percent else monthly
    return df


def create_metadata(df):
    """
    Create metadata dictionary for a standardized asset DataFrame.
    Handles empty DataFrames and NaT index values gracefully.
    """
    start_date = df.index.min()
    end_date = df.index.max()
    metadata = {
        "assets_included": ["Gold", "BTC-USD", "sp500", "10Y bond", "Cash"],
        "source": {"Gold": "https://www.worldbank.org/en/research/commodity-markets",
                   "BTC-USD": "http://www.stooq.com",
                   "sp500": "Robert Shiller's dataset: http://www.econ.yale.edu/~shiller/data.htm",
                   "10Y bond": "Robert Shiller's dataset",
                   "Cash": "Federal Reserve Economic Data (FRED) https://fred.stlouisfed.org/ and Robert Shiller's dataset"},
        "frequency": "monthly",
        "columns": list(df.columns),
        "last_update": datetime.now().strftime("%Y-%m-%d"),
        "start_date": start_date.strftime("%Y-%m-%d") if pd.notnull(start_date) else None,
        "end_date": end_date.strftime("%Y-%m-%d") if pd.notnull(end_date) else None,
        "row_count": len(df)
    }
    return metadata

def save_metadata(metadata, out_path):
    """
    Save metadata dictionary as a JSON file.
    """
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)
