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

    def calculate_correlation(self, window=None):
        """Calculate correlation between the two assets."""
        if window is None:
            # Overall correlation
            corr = self.data.corr().iloc[0, 1]
            print(f"\nOverall Correlation: {corr:.4f}")
        else:
            # Rolling correlation
            corr = self.data[self.tickers[0]].rolling(window).corr(self.data[self.tickers[1]])
            print(f"\nRolling Correlation (window={window}):")
            print(f"  Mean: {corr.mean():.4f}")
            print(f"  Std: {corr.std():.4f}")
            print(f"  Min: {corr.min():.4f}")
            print(f"  Max: {corr.max():.4f}")

        self.results['correlation'] = corr
        return corr

    def engle_granger_test(self):
        """
        Perform Engle-Granger cointegration test.
        Tests if residuals from OLS regression are stationary.
        """
        print("\n" + "=" * 60)
        print("ENGLE-GRANGER COINTEGRATION TEST")
        print("=" * 60)

        # Get prices
        y = self.data[self.tickers[0]].values
        x = self.data[self.tickers[1]].values

        # OLS regression: y = alpha + beta * x
        X = np.column_stack([np.ones(len(x)), x])
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        alpha, beta = coeffs

        # Calculate residuals (spread)
        residuals = y - (alpha + beta * x)

        # ADF test on residuals
        adf_result = adfuller(residuals, autolag='AIC')
        adf_stat, p_value = adf_result[0], adf_result[1]

        # Statsmodels coint function
        coint_stat, coint_pvalue, crit_values = coint(self.data[self.tickers[0]],
                                                      self.data[self.tickers[1]])

        print(f"\nOLS Regression: {self.tickers[0]} = {alpha:.4f} + {beta:.4f} * {self.tickers[1]}")
        print(f"\nADF Test on Residuals:")
        print(f"  ADF Statistic: {adf_stat:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Critical Values: {dict(zip(['1%', '5%', '10%'], adf_result[4].values()))}")

        print(f"\nCointegration Test:")
        print(f"  Test Statistic: {coint_stat:.4f}")
        print(f"  P-value: {coint_pvalue:.4f}")

        is_cointegrated = p_value < 0.05
        print(f"\nResult: {'COINTEGRATED' if is_cointegrated else 'NOT COINTEGRATED'} at 5% significance")

        self.results['engle_granger'] = {
            'alpha': alpha,
            'beta': beta,
            'hedge_ratio': beta,
            'residuals': residuals,
            'adf_statistic': adf_stat,
            'adf_pvalue': p_value,
            'coint_statistic': coint_stat,
            'coint_pvalue': coint_pvalue,
            'is_cointegrated': is_cointegrated
        }

        return self.results['engle_granger']

    def johansen_test(self, det_order=0, k_ar_diff=1):
        """
        Perform Johansen cointegration test.

        Parameters:
        -----------
        det_order : int
            -1: no deterministic term
             0: constant term
             1: linear trend
        k_ar_diff : int
            Number of lagged differences in the model
        """
        print("\n" + "=" * 60)
        print("JOHANSEN COINTEGRATION TEST")
        print("=" * 60)

        # Johansen test
        result = coint_johansen(self.data, det_order=det_order, k_ar_diff=k_ar_diff)

        print(f"\nTrace Statistics:")
        for i, (trace_stat, crit_vals) in enumerate(zip(result.lr1, result.cvt)):
            print(f"  r <= {i}: {trace_stat:.4f} | Critical values (90%, 95%, 99%): {crit_vals}")

        print(f"\nMax Eigenvalue Statistics:")
        for i, (eigen_stat, crit_vals) in enumerate(zip(result.lr2, result.cvm)):
            print(f"  r = {i}: {eigen_stat:.4f} | Critical values (90%, 95%, 99%): {crit_vals}")

        print(f"\nEigenvectors (cointegrating vectors):")
        print(result.evec)

        # Check cointegration at 5% significance
        n_coint = np.sum(result.lr1 > result.cvt[:, 1])  # 95% critical value
        print(f"\nNumber of cointegrating relationships at 5% level: {n_coint}")

        # Extract first eigenvector as hedge ratio
        if n_coint > 0:
            evec = result.evec[:, 0]
            print(f"\nFirst Eigenvector (normalized): {evec}")
            hedge_ratio = -evec[1] / evec[0]
            print(f"Implied Hedge Ratio: {hedge_ratio:.4f}")
        else:
            evec = None
            hedge_ratio = None

        self.results['johansen'] = {
            'trace_stats': result.lr1,
            'eigen_stats': result.lr2,
            'trace_crit': result.cvt,
            'eigen_crit': result.cvm,
            'eigenvectors': result.evec,
            'eigenvalues': result.eig,
            'n_coint': n_coint,
            'hedge_ratio': hedge_ratio
        }

        return self.results['johansen']

    def plot_prices(self, save_path='reports/figures/price_series.png'):
        """Plot price series for both assets."""
        fig, ax = plt.subplots(figsize=(12, 6))

        for ticker in self.tickers:
            ax.plot(self.data.index, self.data[ticker], label=ticker, linewidth=1.5)

        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.set_title('Price Series: KO vs PEP')
        ax.legend()
        ax.grid(True, alpha=0.3)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[OK] Price plot saved to {save_path}")
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
        ax1.axhline(y=np.mean(residuals) + 2 * np.std(residuals), color='green',
                    linestyle='--', linewidth=1, label='+2σ')
        ax1.axhline(y=np.mean(residuals) - 2 * np.std(residuals), color='green',
                    linestyle='--', linewidth=1, label='-2σ')
        ax1.set_ylabel('Spread')
        ax1.set_title('Spread Evolution (Engle-Granger)')
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
        print(f"[OK] Spread plot saved to {save_path}")
        plt.close()


if __name__ == "__main__":
    # Test the pair selector
    from data_loader import DataLoader

    print("Loading data...")
    loader = DataLoader(tickers=['KO', 'PEP'])
    data = loader.load_data()

    print("\nAnalyzing pair...")
    selector = PairSelector(data)

    # Calculate correlation
    selector.calculate_correlation()

    # Engle-Granger test
    eg_results = selector.engle_granger_test()

    # Johansen test
    johansen_results = selector.johansen_test()

    # Generate plots
    selector.plot_prices()
    selector.plot_spread()
