"""
visualize.py

Plotting utilities for WealthPathAnalyzer.
Includes:
- Wealth path and time series plotting functions

Author: Fatima-Ezzahra Jabiri
Date: 2026-02-03
"""
import matplotlib.pyplot as plt


def plot_wealth_path(wealth_df, columns, labels, title= ""):
    
    labels_keys = list(labels.keys())

    plt.figure(figsize=(10,6))
    for col  in columns: 
        if col not in labels_keys:
            lab = f"{col}_path"
        else: 
            lab = labels[col]
        plt.plot(wealth_df.index, wealth_df[col], label = lab)
    
    plt.xlabel("Date")
    plt.ylabel("Wealth ($)")
    plt.title(f"{title}")
    plt.legend()
    plt.grid()
    plt.show();