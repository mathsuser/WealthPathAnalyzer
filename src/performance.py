"""
performance.py


Portfolio and asset risk/return metrics for WealthPathAnalyzer.

Includes:
- CAGR, volatility, Sharpe ratio, skewness, kurtosis
- Drawdown, max drawdown
- Value at Risk (VaR), Conditional VaR (CVaR)
- performance_summary module that renders all the risk/returns metrics 

Author: Fatima-Ezzahra Jabiri
Date: 2026-02-04
"""
import pandas as pd
import numpy as np
from scipy.stats import skew as scipy_skew
from scipy.stats import kurtosis as scipy_kurtosis



def calculate_cagr(returns, periods_per_year: int = 12):
    """
    Computes the Compound Annual Growth Rate (CAGR) of a return series od DataFrame of returns.

    This represents the annualized geometric return, assuming compounding.
    For a 1-year horizon (12 periods), it reduces to the geometric average:
        1 + CAGR = product(1 + r_i), for monthly r_i over 12 months

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic total returns (simple returns), e.g., monthly portfolio returns.
    periods_per_year : int, default 12
        Number of periods per year.

    Returns
    -------
    float
        Compound annual growth rate. Returns NaN if invalid (e.g., non-positive gross path).

    Important
    ---------
    - Input must be TOTAL returns, not excess returns. 
    """
    if isinstance(returns, pd.DataFrame):
        def _cagr(col):
            r = col.dropna()
            if r.empty:
                return np.nan
            gross = 1.0 + r
            compounded = gross.prod()
            n = len(r)
            return compounded ** (periods_per_year / n) - 1.0
        return returns.apply(_cagr, axis=0)
    else:
        r = returns.dropna()
        if r.empty:
            return np.nan
        gross = 1.0 + r
        compounded = gross.prod()
        n = len(r)
        return compounded ** (periods_per_year / n) - 1.0

def calculate_volatility(returns, periods_per_year = 12, ddof =1):
    """
    Calculate annualised volatility of a time series (single asset or portfolio).
    """
    
    r = returns.copy()
    if r.empty:
        return np.nan
    if isinstance(returns, pd.DataFrame):
        return r.apply(lambda x: x.dropna().std(ddof=ddof) * np.sqrt(periods_per_year))
    return r.dropna().std(ddof=ddof) * np.sqrt(periods_per_year)


def sharpe_ratio(returns, risk_free=None, periods_per_year=12, ddof=1):
    """
    Calculate the annualized Sharpe ratio for a return series or DataFrame.
    If risk_free is None, computes mean(returns) / std(returns).
    If risk_free is a float, subtracts it from returns before calculation.
    If risk_free is a Series/DataFrame, aligns and subtracts it from returns.
    Assumes risk_free is in the same periodicity/units as returns.
    """
    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda col: sharpe_ratio(col, risk_free[col.name] if isinstance(risk_free, (pd.Series, pd.DataFrame)) and col.name in risk_free else risk_free, periods_per_year, ddof), axis=0)

    r = returns.dropna()
    if r.empty:
        return np.nan
    if risk_free is None:
        excess = r
    elif isinstance(risk_free, (float, int)):
        excess = r - risk_free
    elif isinstance(risk_free, (pd.Series, pd.DataFrame)):
        excess = r - risk_free.reindex_like(r)
    else:
        raise TypeError("risk_free must be None, float, Series, or DataFrame")
    mean_excess = excess.mean() * periods_per_year
    vol = excess.std(ddof=ddof) * np.sqrt(periods_per_year)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return mean_excess / vol



def skewness(returns):
    """
    Robust skewness using scipy.stats.skew (bias-corrected, matches industry standard).
    Returns a float for Series, or Series for DataFrame.
    """
    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda x: scipy_skew(x.dropna(), bias=False), axis=0)
    else:
        return scipy_skew(pd.Series(returns).dropna(), bias=False)



def kurtosis(returns):
    """
    Robust kurtosis using scipy.stats.kurtosis (Fisher definition, bias-corrected).
    Returns a float for Series, or Series for DataFrame.
    """
    if isinstance(returns, pd.DataFrame):
        # Add 3 to get Pearson kurtosis (normal=3)
        return returns.apply(lambda x: scipy_kurtosis(x.dropna(), fisher=True, bias=False) + 3, axis=0)
    else:
        return scipy_kurtosis(pd.Series(returns).dropna(), fisher=True, bias=False) + 3

def calculate_drawdown(returns: pd.Series) -> pd.DataFrame:
    """
    Compute wealth index, previous peaks, and drawdowns from periodic returns.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns.

    Returns
    -------
    pd.DataFrame
        Columns:
        - wealth_index : cumulative wealth assuming initial 1.0
        - previous_peaks: running maximum of wealth_index
        - drawdown     : drawdown series (wealth_index / previous_peaks - 1)
    """
    r = pd.Series(returns).dropna()
    if r.empty:
        return pd.DataFrame(columns=["wealth_index", "previous_peaks", "drawdown"])
    wealth = (1.0 + r).cumprod()
    peaks = wealth.cummax()
    dd = wealth / peaks - 1.0
    return pd.DataFrame({"wealth_index": wealth, "previous_peaks": peaks, "drawdown": dd})


def max_drawdown(returns):
    """
    Compute the maximum drawdown from a return series or DataFrame of returns.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic returns.

    Returns
    -------
    float or pd.Series
        Minimum value of the drawdown series (a negative number). NaN if undefined.
    """
    if isinstance(returns, pd.DataFrame):
        return returns.apply(max_drawdown, axis=0)
    dd = calculate_drawdown(returns)
    if dd.empty:
        return np.nan
    return float(dd["drawdown"].min())



def var_historic(returns, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(returns, pd.DataFrame):
        return returns.aggregate(var_historic, level=level)
    elif isinstance(returns, pd.Series):
        returns = returns.dropna()
        if returns.empty:
            return np.nan
        return -np.percentile(returns, level)
    else:
        raise TypeError("Expected returns to be a Series or DataFrame")


def cvar_historic(returns, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna()
        if returns.empty:
            return np.nan
        var_val = var_historic(returns, level=level)
        is_beyond = returns <= -var_val if np.isfinite(var_val) else False
        if not np.any(is_beyond):
            return np.nan
        return -returns[is_beyond].mean()
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected returns to be a Series or DataFrame")

def performance_summary(returns, name=None, risk_free=None, periods_per_year=12):
    """
    Compute a summary DataFrame of key performance metrics for one or more return series.
    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Return series (can be one or many).
    name : str or list, optional
        Name(s) for the strategy/asset(s). Used as index in output.
    risk_free : None, float, pd.Series, or pd.DataFrame, optional
        Risk-free rate for Sharpe ratio.
    periods_per_year : int, default 12
        Number of periods per year (for annualization).
    Returns
    -------
    pd.DataFrame
        DataFrame with metrics as columns and assets/strategies as index.
    """
    metrics = {}
    metrics['CAGR'] = calculate_cagr(returns, periods_per_year)
    metrics['Volatility'] = calculate_volatility(returns, periods_per_year)
    metrics['Sharpe'] = sharpe_ratio(returns, risk_free, periods_per_year)
    metrics['Max Drawdown'] = max_drawdown(returns)
    metrics['VaR (5%)'] = var_historic(returns, level=5)
    metrics['CVaR (5%)'] = cvar_historic(returns, level=5)
    metrics['Skew'] = skewness(returns)
    metrics['Kurtosis'] = kurtosis(returns)
    if isinstance(returns, pd.Series):
        idx = [name] if name is not None else [returns.name if returns.name is not None else 'Strategy']
        df = pd.DataFrame([metrics], index=idx)
    else:
        df = pd.DataFrame(metrics)
        if name is not None:
            df.index = name if isinstance(name, list) else [name]
    return df.round(4)