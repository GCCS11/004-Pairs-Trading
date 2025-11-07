"""
Data Preprocessing Module
Splits data into train/test/validation sets and prepares features.
"""

import pandas as pd
import numpy as np
from pathlib import Path


class DataPreprocessor:
    """Handles data splitting and preprocessing for pairs trading."""

    def __init__(self, data, train_ratio=0.6, test_ratio=0.2, val_ratio=0.2):
        """
        Initialize preprocessor.

        Parameters:
        -----------
        data : pd.DataFrame
            Price data for the pair
        train_ratio : float
            Proportion for training (default 0.6)
        test_ratio : float
            Proportion for testing (default 0.2)
        val_ratio : float
            Proportion for validation (default 0.2)
        """
        if not np.isclose(train_ratio + test_ratio + val_ratio, 1.0):
            raise ValueError("Ratios must sum to 1.0")

        self.data = data
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio

        self.train_data = None
        self.test_data = None
        self.val_data = None

    def split_data(self):
        """Split data chronologically into train/test/validation sets."""
        n = len(self.data)

        train_end = int(n * self.train_ratio)
        test_end = int(n * (self.train_ratio + self.test_ratio))

        self.train_data = self.data.iloc[:train_end].copy()
        self.test_data = self.data.iloc[train_end:test_end].copy()
        self.val_data = self.data.iloc[test_end:].copy()

        print(
            f"\nTrain: {len(self.train_data)} days ({self.train_data.index[0].date()} to {self.train_data.index[-1].date()})")
        print(
            f"Test: {len(self.test_data)} days ({self.test_data.index[0].date()} to {self.test_data.index[-1].date()})")
        print(f"Val: {len(self.val_data)} days ({self.val_data.index[0].date()} to {self.val_data.index[-1].date()})")

        return self.train_data, self.test_data, self.val_data

    def save_splits(self, data_dir='data/processed'):
        """Save train/test/validation splits to CSV files."""
        if self.train_data is None:
            raise ValueError("Run split_data() first")

        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)

        self.train_data.to_csv(data_path / 'train.csv')
        self.test_data.to_csv(data_path / 'test.csv')
        self.val_data.to_csv(data_path / 'val.csv')

    def load_splits(self, data_dir='data/processed'):
        """Load previously saved splits."""
        data_path = Path(data_dir)

        self.train_data = pd.read_csv(data_path / 'train.csv', index_col='Date', parse_dates=True)
        self.test_data = pd.read_csv(data_path / 'test.csv', index_col='Date', parse_dates=True)
        self.val_data = pd.read_csv(data_path / 'val.csv', index_col='Date', parse_dates=True)

        return self.train_data, self.test_data, self.val_data


if __name__ == "__main__":
    from data_loader import DataLoader

    loader = DataLoader(tickers=['HD', 'LOW'])
    data = loader.load_data()

    preprocessor = DataPreprocessor(data)
    train, test, val = preprocessor.split_data()
    preprocessor.save_splits()
