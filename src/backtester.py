

import pandas as pd
from src.returns import *

# === Wealth Path calculations  ===

def compute_wealth_path(df, index_col, initial_investment=1.0, cpi = None, asset_name = "asset", real_wealth = False, cash_col = None, is_accumulated = False):
    """
    calculates nominal wealth path, and real wealth path when cpi series is provided. 
    """
    
    try: 
        wealth_path = initial_investment * df[f"{index_col}"]
    except Exception as e:
            raise ValueError(f"{index_col} column is missing or invalid in the DataFrame.")
    
    wealth_df = pd.DataFrame({
        f"{asset_name}_nominal_wealth": wealth_path
    }, index = df.index)


    if cash_col:
        if  is_accumulated:
            
            if asset_name == "10Y bond":
                initial_price = 1.0
                shares = initial_investment / initial_price
                wealth_df.loc[:, f"{asset_name}_Accumulated_Coupon"] = df[cash_col].fillna(0).cumsum() * shares
                cash_series = wealth_df[f"{asset_name}_Accumulated_Coupon"]
            elif asset_name == "sp500":
                initial_price = df["sp500_Price"].iloc[0]
                shares = initial_investment / initial_price
                wealth_df.loc[:, f"{asset_name}_Accumulated_Dividend"] = (df[cash_col].fillna(0).cumsum() / 12) * shares
                cash_series = wealth_df[f"{asset_name}_Accumulated_Dividend"]
            else:
                raise ValueError("cash_col' is only supported for '10Y bond' and 'sp500' assets.")
            
            accumulated_cash = cash_series
            total_wealth = wealth_path + accumulated_cash
            wealth_df.loc[:, f"{asset_name}_price_wealth"] = wealth_path
            wealth_df.loc[:, f"{asset_name}_nominal_wealth"] = total_wealth
        else:
            # For informational purposes, display the raw cashflow column scaled to investment (not just per share/unit)
            if asset_name == "10Y bond":
                shares = initial_investment / 1.0
            elif asset_name == "sp500":
                shares = initial_investment / df["sp500_Price"].iloc[0]
            else:
                shares = 1.0
            wealth_df.loc[:, "Raw_Cashflow_Info"] = df[cash_col].fillna(0) * shares
            # Wealth is price-only
            wealth_df.loc[:, f"{asset_name}_price_wealth"] = wealth_path
            wealth_df.loc[:, f"{asset_name}_nominal_wealth"] = wealth_path
    else:
        total_wealth = wealth_path

    if real_wealth:
        if cpi is None:
            raise ValueError("CPI data required for real wealth calculation. If non avaliable, turn off real_wealth")

        # Case of cpi
        wealth_df.loc[:, "CPI"] = cpi.reindex(df.index, method="ffill")
        
        inflation_factor = cpi/cpi.iloc[-1]
        real_wealth = inflation_factor * wealth_path
        wealth_df.loc[:, f"{asset_name}_real_wealth"] = real_wealth

        if cash_col:
            if not is_accumulated:
                # Use already calculated nominal accumulated cashflows and adjust for inflation
                if asset_name == "10Y bond":
                    # Real accumulated coupon
                    wealth_df.loc[:, f"{asset_name}_Real_Accumulated_Coupon"] = wealth_df[f"{asset_name}_Accumulated_Coupon"] * inflation_factor
                    shares = initial_investment / 1.0
                    wealth_df.loc[:, f"{asset_name}_total_real_wealth"] = (initial_investment * inflation_factor * df[index_col]) + (wealth_df[f"{asset_name}_Real_Accumulated_Coupon"] * shares)
                elif asset_name == "sp500":
                    # Real accumulated dividend (annualized)
                    wealth_df.loc[:, f"{asset_name}_Real_Accumulated_Dividend"] = wealth_df[f"{asset_name}_Accumulated_Dividend"] * inflation_factor
                    shares = initial_investment / df["sp500_Price"].iloc[0]
                    wealth_df.loc[:, f"{asset_name}_total_real_wealth"] = (initial_investment * inflation_factor * df[index_col]) + (wealth_df[f"{asset_name}_Real_Accumulated_Dividend"] * shares)
                # else: already handled above



    return wealth_df    

# === single-asset buy-and-hold strategy ===
CORE_ASSETS = ["Cash", "Gold", "sp500", "10Y bond", "BTC-USD"]
def single_name_backtester(backtest_df,
        start_date=None, end_date=None, 
        initial_investment = 1.0, asset_name = "asset",  
        real_wealth = False, cpi=None, is_accumulated = False):
    

    if asset_name not in CORE_ASSETS:
        raise ValueError(f"{asset_name} not supported. Choose name within {CORE_ASSETS}")
    
    # Step 1: Trim the data frame for the backtest period
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df = backtest_df.loc[start_date:end_date].copy()
    cpi_df = None
    if cpi is not None: 
        cpi_df = cpi.loc[start_date:end_date].copy()


    # Step 2: Use the calculate index functions in returns.py to calculate the returns index
    #         Then the compute_wealth_index to calculate the wealth path            
    if (asset_name=="Gold") or (asset_name=="BTC-USD"):
        # Call calculate_price_index: 
        df = build_price_index(df, return_col=f"{asset_name}_Return", asset_name=asset_name)
        # call compute_wealth_path:
        wealth_df = compute_wealth_path(df, index_col=f"{asset_name}_Price_Index", initial_investment= initial_investment, cpi= cpi_df,asset_name=asset_name, real_wealth = True)
        # add the price index column
        wealth_df[f"{asset_name}_Price_Index"] = df[f"{asset_name}_Price_Index"]


    elif asset_name == "Cash":
        df = build_total_return_index(df, return_col="Cash_Monthly_Rate (decimal)", asset_name=asset_name)
        # call compute_wealth_path:
        wealth_df = compute_wealth_path(df, index_col=f"{asset_name}_TR_Index", initial_investment= initial_investment, cpi= cpi_df,asset_name=asset_name, real_wealth = True)
        # add the price index column
        wealth_df[f"{asset_name}_TR_Index"] = df[f"{asset_name}_TR_Index"] 

    else :
        df = build_price_index(df, return_col=f"{asset_name}_Price_Return", asset_name=asset_name)
        df = build_total_return_index(df, return_col=f"{asset_name}_Total_Return", asset_name=asset_name)

        # Default: do NOT show accumulated cashflows (cash_col=None)
        cash_col = None
        if is_accumulated:
            if asset_name == "sp500":
                cash_col = f"{asset_name}_Dividend"
            else:
                cash_col = f"{asset_name}_Coupon_Income"
            wealth_df = compute_wealth_path(
                df,
                index_col=f"{asset_name}_Price_Index",
                initial_investment=initial_investment,
                cpi=cpi_df,
                asset_name=asset_name,
                real_wealth=real_wealth,
                cash_col=cash_col,
                is_accumulated=is_accumulated
            )
            wealth_df[f"{asset_name}_Price_Index"] = df[f"{asset_name}_Price_Index"]
        else:
            wealth_df = compute_wealth_path(
                df,
                index_col=f"{asset_name}_TR_Index",
                initial_investment=initial_investment,
                cpi=cpi_df,
                asset_name=asset_name,
                real_wealth=real_wealth,
                cash_col=None,
                is_accumulated=is_accumulated
            )
            wealth_df[f"{asset_name}_TR_Index"] = df[f"{asset_name}_TR_Index"]

    
    return wealth_df

    
