"""
Data Loader Module
Downloads and prepares historical price data for pairs trading analysis.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


class DataLoader:
    """Downloads and manages historical stock price data."""

    def __init__(self, tickers, start_date=None, end_date=None, data_dir='data/raw'):
        """
        Initialize DataLoader.

        Parameters:
        -----------
        tickers : list
            List of ticker symbols (e.g., ['HD', 'LOW'])
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        data_dir : str
            Directory to save raw data
        """
        self.tickers = tickers
        self.start_date = start_date or self._get_default_start_date()
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data = None

    def _get_default_start_date(self):
        """Get default start date (15 years ago)."""
        default_start = datetime.now() - timedelta(days=15*365)
        return default_start.strftime('%Y-%m-%d')

    def download_data(self):
        """Download historical data from Yahoo Finance."""
        all_data = []

        for ticker in self.tickers:
            df = yf.download(ticker, start=self.start_date, end=self.end_date,
                           progress=False, auto_adjust=False)

            if df.empty:
                raise ValueError(f"No data downloaded for {ticker}")

            price_series = df['Adj Close']
            price_series.name = ticker
            all_data.append(price_series)

        self.data = pd.concat(all_data, axis=1)
        self.data.index.name = 'Date'
        self.data = self.data.ffill().bfill()

        print(f"Downloaded {len(self.data)} days ({self.data.index[0].date()} to {self.data.index[-1].date()})")
        return self.data

    def save_data(self, filename='pair_prices.csv'):
        """Save data to CSV."""
        if self.data is None:
            raise ValueError("No data to save. Run download_data() first.")

        filepath = self.data_dir / filename
        self.data.to_csv(filepath)
        return filepath

    def load_data(self, filename='pair_prices.csv'):
        """Load data from CSV."""
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        self.data = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        return self.data

    def get_summary_stats(self):
        """Get summary statistics of the data."""
        if self.data is None:
            raise ValueError("No data loaded.")

        print(f"\nData: {self.data.shape[0]} days, {self.data.shape[1]} assets")
        print(f"Period: {self.data.index[0].date()} to {self.data.index[-1].date()}")
        print(f"\n{self.data.describe()}")

        return self.data.describe()


if __name__ == "__main__":
    loader = DataLoader(tickers=['HD', 'LOW'])
    data = loader.download_data()
    loader.save_data()
    loader.get_summary_stats()