"""
returns.py

Core asset returns and wealth calculation functions for WealthPathAnalyzer.

Includes:
- Price and total return calculations for equities, bonds, gold, BTC, and cash
- Rolling bond strategy (Shiller method)
- Dividend/coupon accumulation and reinvestment logic
- Price and total return index construction
- Wealth path and inflation adjustment utilities

Author: Fatima-Ezzahra Jabiri
Date: 2026-02-01
"""

import pandas as pd
import numpy as np


# === Returns Calculations for Price appreciation only assets ===

def compute_price_returns(df, price_col, asset_name="asset"):
    """
    Calculates monthly price returns for gold and bitcoin. 
    asset_name: "Gold" or "BTC-USD"

    """
    try:
        df[f"{asset_name}_Return"] = df[price_col].pct_change()
        return df
    except Exception as e:
        raise ValueError (f"{price_col} column is missing or invalid in the DataFrame.") from e
    

# === Returns calculations for sp500-like assets (dividend paying) ===

def compute_equity_returns(df, price_col, dividend_col, asset_name="sp500"):
    """
    Calculates : 
    - monthly total returns for equity assets with dividends; 
    - price appreciation only returns and 
    - price appreciation with accumulated dividend as cash.
    parameters:
    df : pd.DataFrame
        DataFrame with price and dividend columns
        dividends are the monthly dividend amounts (not yield)
    """

    if (dividend_col not in df.columns) or (price_col not in df.columns):
        raise ValueError(f"Either {dividend_col} or {price_col} column is missing in the DataFrame.")
    else:
        # Calculate price returns
        df.loc[:, f"{asset_name}_Price_Return"] = df[price_col].pct_change()
        # Calculate total returns (price + dividends) - divid by 12 as dividends are annual
        df.loc[:, f"{asset_name}_Total_Return"] = ((df[price_col] + df[dividend_col]/12) / df[price_col].shift(1) - 1)
        # Track accumulated dividends separately
        df.loc[:, f"{asset_name}_Accumulated_Dividend"] = (df[dividend_col].fillna(0).cumsum() / 12)

        return df
    
# === Returns calculation for treasury-like assets (bond rolling strategy) ===

def compute_bond_returns(df, yield_col = "10Y_Treasury_Annual_Rate (%)", asset_name="10Y bond"):
    """
    Calculates monthly total returns for bond assets using Shiller's rolling strategy.
    parameters:
    df : pd.DataFrame
        DataFrame with yield column annual in percentage 
    """
    if yield_col not in df.columns:
        raise ValueError(f"{yield_col} column is missing in the DataFrame.")
    else:
        # Initialize core data for Shiller's method
        yields = df[yield_col]
        bond_total_returns = []
        bond_price_changes = []
        bond_prices = []  # Starting at par
        maturity_years = 10
        # === Core Shiller Bond Calculations (Same for all strategies) ===
        # Calculate price changes and coupon income using Shiller's method
        for t in range(0, len(yields) - 1):
            y_t = yields.iloc[t]      # Current yield
            y_t1 = yields.iloc[t + 1] # Next period yield
            r = y_t1 / 1200                  # Discount rate
            c = y_t / 1200                  # Coupon rate (monthly)

            # Standard Shiller bond pricing
            months_remaining = maturity_years * 12 - 1
            if r == 0:
                price_t1 = c * months_remaining + 1
            else:
                annuity_factor = (1 - (1 + r) ** -months_remaining) / r
                price_t1 = c * annuity_factor + 1 / (1 + r) ** months_remaining

            # Store components for different strategies
            price_change = price_t1 - 1.0  # Price return component
            coupon_income = c              # Income component
            
            
            R_t = price_change + coupon_income
                
            bond_total_returns.append(R_t)
            bond_price_changes.append(price_change)
            bond_prices.append(price_t1)
        # Complete the series
        bond_total_returns.append(np.nan)
        bond_price_changes.append(np.nan)
        bond_prices.append(np.nan)
        df.loc[:, f"{asset_name}_Total_Return"] = pd.Series(bond_total_returns, index=df.index)
        df.loc[:, f"{asset_name}_Price_Return"] = pd.Series(bond_price_changes, index=df.index)
        df.loc[:, f"{asset_name}_Price"] = pd.Series(bond_prices, index=df.index)

        # Add coupon income column and accumulated coupon income
        df.loc[:, f"{asset_name}_Coupon_Income"] = (df[yield_col] / 1200)
        df.loc[:, f"{asset_name}_Accumulated_Coupon"] = df[f"{asset_name}_Coupon_Income"].cumsum()


        return df



# === Price Index calculations  ===

def build_price_index(df, return_col, asset_name="asset"):
    """
    Build price index from returns series.
    - For Gold: asset_name = "Gold"
    - For Bitcoin: asset_name="BTC-USD"
    - For sp500: asset_name = "sp500"
    """

    try:
        price_index = (1 + df[f"{return_col}"]).cumprod()
        df.loc[:, f"{asset_name}_Price_Index"] = price_index
        df.loc[df.index[0], f"{asset_name}_Price_Index"] = 1.0  # Set first value to 1.0
    except Exception as e:
        raise ValueError(f"{return_col} column is missing or invalid in the DataFrame.")
    
    return df

def build_total_return_index(df, return_col, asset_name="asset"):
    """
    Build total returns index from total returns series. 

    """    
    try:
        TR_index = (1 + df[f"{return_col}"]).cumprod()
        df.loc[:, f"{asset_name}_TR_Index"] = TR_index
        # replace only if it is nan
        if pd.isna(df.loc[df.index[0], f"{asset_name}_TR_Index"]):
            df.loc[df.index[0], f"{asset_name}_TR_Index"] = 1.0
    except Exception as e:
        raise ValueError(f"{return_col} column is missing or invalid in the DataFrame.")
    return df


