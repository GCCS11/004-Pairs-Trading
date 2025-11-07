"""
Pair Selection Module
Tests for correlation and cointegration between asset pairs.
Implements both Engle-Granger and Johansen cointegration tests.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class PairSelector:
    """Analyzes and selects cointegrated pairs for trading."""

    def __init__(self, data):
        """
        Initialize PairSelector.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with price data for two assets
        """
        self.data = data
        self.tickers = list(data.columns)
        self.results = {}

        if len(self.tickers) != 2:
            raise ValueError("PairSelector requires exactly 2 tickers")

    def calculate_correlation(self):
        """Calculate correlation between the two assets."""
        corr = self.data.corr().iloc[0, 1]
        print(f"Correlation: {corr:.4f}")
        self.results['correlation'] = corr
        return corr

    def engle_granger_test(self):
        """Perform Engle-Granger cointegration test."""
        y = self.data[self.tickers[0]].values
        x = self.data[self.tickers[1]].values

        # OLS regression
        X = np.column_stack([np.ones(len(x)), x])
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        alpha, beta = coeffs

        residuals = y - (alpha + beta * x)

        # ADF test on residuals
        adf_result = adfuller(residuals, autolag='AIC')
        adf_stat, p_value = adf_result[0], adf_result[1]

        is_cointegrated = p_value < 0.05

        print(f"\nEngle-Granger: {self.tickers[0]} = {alpha:.4f} + {beta:.4f} * {self.tickers[1]}")
        print(f"  ADF p-value: {p_value:.4f} - {'COINTEGRATED' if is_cointegrated else 'NOT COINTEGRATED'}")

        self.results['engle_granger'] = {
            'alpha': alpha,
            'beta': beta,
            'hedge_ratio': beta,
            'residuals': residuals,
            'adf_statistic': adf_stat,
            'adf_pvalue': p_value,
            'is_cointegrated': is_cointegrated
        }

        return self.results['engle_granger']

    def johansen_test(self, det_order=0, k_ar_diff=1):
        """Perform Johansen cointegration test."""
        result = coint_johansen(self.data, det_order=det_order, k_ar_diff=k_ar_diff)

        # Check cointegration at 5% significance
        n_coint = np.sum(result.lr1 > result.cvt[:, 1])

        print(f"\nJohansen Test:")
        print(f"  Trace stat: {result.lr1[0]:.4f} (critical 5%: {result.cvt[0, 1]:.4f})")
        print(f"  Cointegrating relationships: {n_coint}")

        if n_coint > 0:
            evec = result.evec[:, 0]
            print(f"  First eigenvector: [{evec[0]:.4f}, {evec[1]:.4f}]")
        else:
            evec = result.evec[:, 0]
            print(f"  Using first eigenvector anyway: [{evec[0]:.4f}, {evec[1]:.4f}]")

        self.results['johansen'] = {
            'trace_stats': result.lr1,
            'eigen_stats': result.lr2,
            'trace_crit': result.cvt,
            'eigen_crit': result.cvm,
            'eigenvectors': result.evec,
            'eigenvalues': result.eig,
            'n_coint': n_coint
        }

        return self.results['johansen']

    def plot_prices(self, save_path='reports/figures/price_series.png'):
        """Plot price series for both assets."""
        fig, ax = plt.subplots(figsize=(12, 6))

        for ticker in self.tickers:
            ax.plot(self.data.index, self.data[ticker], label=ticker, linewidth=1.5)

        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        tickers_str = ' vs '.join(self.tickers)
        ax.set_title(f'Price Series: {tickers_str}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_spread(self, save_path='reports/figures/spread_evolution.png'):
        """Plot the spread from Engle-Granger method."""
        if 'engle_granger' not in self.results:
            raise ValueError("Run engle_granger_test() first")

        residuals = self.results['engle_granger']['residuals']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Spread over time
        ax1.plot(self.data.index, residuals, linewidth=1, color='darkblue')
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax1.axhline(y=np.mean(residuals) + 2*np.std(residuals), color='green',
                   linestyle='--', linewidth=1, label='+2σ')
        ax1.axhline(y=np.mean(residuals) - 2*np.std(residuals), color='green',
                   linestyle='--', linewidth=1, label='-2σ')
        ax1.set_ylabel('Spread')
        tickers_str = f"{self.tickers[0]}-{self.tickers[1]}"
        ax1.set_title(f'Spread Evolution: {tickers_str} (Engle-Granger)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Distribution
        ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Spread Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Spread Distribution')
        ax2.grid(True, alpha=0.3)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    from data_loader import DataLoader

    loader = DataLoader(tickers=['HD', 'LOW'])
    data = loader.load_data()

    selector = PairSelector(data)
    selector.calculate_correlation()
    selector.engle_granger_test()
    selector.johansen_test()
    selector.plot_prices()
    selector.plot_spread()