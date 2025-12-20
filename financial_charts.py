"""
Financial Charting Utilities
Alternative to mplfinance for Python 3.13 compatibility
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def setup_matplotlib_for_plotting():
    """Setup matplotlib for plotting with proper configuration."""
    plt.switch_backend("Agg")
    plt.style.use("seaborn-v0_8")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

def plot_candlestick(df, title="Candlestick Chart", figsize=(12, 8), save_path=None):
    """
    Create a candlestick chart
    
    Args:
        df: DataFrame with 'Open', 'High', 'Low', 'Close' columns
        title: Chart title
        figsize: Figure size (width, height)
        save_path: Path to save the chart
    """
    setup_matplotlib_for_plotting()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for idx, row in df.iterrows():
        date = mdates.date2num(idx) if isinstance(idx, datetime) else idx
        open_price = row['Open']
        high_price = row['High']
        low_price = row['Low']
        close_price = row['Close']
        
        # Determine color (green for up, red for down)
        color = 'green' if close_price >= open_price else 'red'
        
        # Draw the high-low line
        ax.plot([date, date], [low_price, high_price], color='black', linewidth=1)
        
        # Draw the open-close rectangle
        height = abs(close_price - open_price)
        bottom = min(open_price, close_price)
        
        ax.bar(date, height, bottom=bottom, width=0.6, 
               color=color, alpha=0.8, edgecolor='black')
    
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    
    # Format x-axis dates
    if hasattr(df.index, 'name') and 'date' in df.index.name.lower():
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(df)//10)))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_line_chart(df, columns=None, title="Line Chart", figsize=(12, 8), save_path=None):
    """
    Create a line chart for multiple columns
    
    Args:
        df: DataFrame with data
        columns: List of columns to plot (default: all numeric columns)
        title: Chart title
        figsize: Figure size
        save_path: Path to save the chart
    """
    setup_matplotlib_for_plotting()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for col in columns:
        ax.plot(df.index, df[col], label=col, linewidth=2)
    
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates if index is datetime
    if isinstance(df.index, pd.DatetimeIndex):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_volume_chart(df, title="Volume Chart", figsize=(12, 4), save_path=None):
    """
    Create a volume bar chart
    
    Args:
        df: DataFrame with 'Volume' column
        title: Chart title
        figsize: Figure size
        save_path: Path to save the chart
    """
    setup_matplotlib_for_plotting()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['green' if df.iloc[i]['Close'] >= df.iloc[i]['Open'] else 'red' 
              for i in range(len(df))]
    
    ax.bar(df.index, df['Volume'], color=colors, alpha=0.7)
    
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates if index is datetime
    if isinstance(df.index, pd.DatetimeIndex):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_ohlcv_chart(df, title="OHLCV Chart", figsize=(15, 10), save_path=None):
    """
    Create a comprehensive OHLCV chart with subplots
    
    Args:
        df: DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume' columns
        title: Chart title
        figsize: Figure size
        save_path: Path to save the chart
    """
    setup_matplotlib_for_plotting()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Price chart
    for idx, row in df.iterrows():
        date = mdates.date2num(idx) if isinstance(idx, datetime) else idx
        open_price = row['Open']
        high_price = row['High']
        low_price = row['Low']
        close_price = row['Close']
        
        color = 'green' if close_price >= open_price else 'red'
        
        # High-low line
        ax1.plot([date, date], [low_price, high_price], color='black', linewidth=1)
        
        # Open-close rectangle
        height = abs(close_price - open_price)
        bottom = min(open_price, close_price)
        
        ax1.bar(date, height, bottom=bottom, width=0.6, 
                color=color, alpha=0.8, edgecolor='black')
    
    ax1.set_title(title)
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    
    # Volume chart
    colors = ['green' if df.iloc[i]['Close'] >= df.iloc[i]['Open'] else 'red' 
              for i in range(len(df))]
    
    ax2.bar(df.index, df['Volume'], color=colors, alpha=0.7)
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis dates
    if isinstance(df.index, pd.DatetimeIndex):
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, (ax1, ax2)

# Mock mplfinance functions for compatibility
def make_addplot(*args, **kwargs):
    """Mock function to replace mplfinance.make_addplot"""
    pass

# Alias for compatibility
plot_candlestick = plot_candlestick
plot_ohlcv = plot_ohlcv_chart

if __name__ == "__main__":
    # Test the module
    print("Financial charting utilities loaded successfully!")
    print("Available functions:")
    print("- plot_candlestick(df, title, figsize, save_path)")
    print("- plot_ohlcv_chart(df, title, figsize, save_path)")
    print("- plot_line_chart(df, columns, title, figsize, save_path)")
    print("- plot_volume_chart(df, title, figsize, save_path)")
