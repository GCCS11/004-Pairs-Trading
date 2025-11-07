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
            List of ticker symbols (e.g., ['KO', 'PEP'])
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
        default_start = datetime.now() - timedelta(days=15 * 365)
        return default_start.strftime('%Y-%m-%d')

    def download_data(self):
        """Download historical data from Yahoo Finance."""
        print(f"Downloading data for {self.tickers}")
        print(f"Period: {self.start_date} to {self.end_date}")

        all_data = []

        for ticker in self.tickers:
            print(f"\nDownloading {ticker}...")
            df = yf.download(ticker, start=self.start_date, end=self.end_date,
                             progress=False, auto_adjust=False)

            print(f"Downloaded data shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")

            if df.empty:
                raise ValueError(f"No data downloaded for {ticker}")

            # Keep only Adjusted Close and rename column
            price_series = df['Adj Close']
            price_series.name = ticker
            all_data.append(price_series)
            print(f"[OK] {ticker}: {len(price_series)} days of data")
            print(f"Added to list. List now has {len(all_data)} items")

        print(f"\nConcatenating {len(all_data)} series...")

        # Combine into single DataFrame
        self.data = pd.concat(all_data, axis=1)
        self.data.index.name = 'Date'

        # Handle missing values
        missing_before = self.data.isnull().sum()
        print(f"\nMissing values before cleaning:")
        for ticker in self.tickers:
            print(f"  {ticker}: {missing_before[ticker]}")

        self.data = self.data.ffill().bfill()

        missing_after = self.data.isnull().sum()
        print(f"Missing values after cleaning:")
        for ticker in self.tickers:
            print(f"  {ticker}: {missing_after[ticker]}")

        print(f"\n[OK] Downloaded {len(self.data)} days of data for {len(self.tickers)} tickers")
        return self.data

    def save_data(self, filename='pair_prices.csv'):
        """Save data to CSV."""
        if self.data is None:
            raise ValueError("No data to save. Run download_data() first.")

        filepath = self.data_dir / filename
        self.data.to_csv(filepath)
        print(f"\n[OK] Data saved to {filepath}")
        return filepath

    def load_data(self, filename='pair_prices.csv'):
        """Load data from CSV."""
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        self.data = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        print(f"[OK] Loaded data from {filepath}")
        print(f"  Shape: {self.data.shape}")
        print(f"  Period: {self.data.index[0]} to {self.data.index[-1]}")
        return self.data

    def get_summary_stats(self):
        """Get summary statistics of the data."""
        if self.data is None:
            raise ValueError("No data loaded.")

        print("\n" + "=" * 60)
        print("DATA SUMMARY")
        print("=" * 60)
        print(f"\nShape: {self.data.shape}")
        print(f"Period: {self.data.index[0].date()} to {self.data.index[-1].date()}")
        print(f"Total trading days: {len(self.data)}")
        print(f"\nFirst 5 rows:")
        print(self.data.head())
        print(f"\nLast 5 rows:")
        print(self.data.tail())
        print(f"\nDescriptive Statistics:")
        print(self.data.describe())
        print("=" * 60)

        return self.data.describe()


if __name__ == "__main__":
    # Test the data loader
    print("Testing DataLoader with HD and LOW...")

    loader = DataLoader(tickers=['HD', 'LOW'])
    data = loader.download_data()
    loader.save_data()
    stats = loader.get_summary_stats()