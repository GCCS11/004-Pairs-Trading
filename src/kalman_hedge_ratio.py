"""
Kalman Filter for Dynamic Hedge Ratio Estimation
Implements KF as a Sequential Decision Process following Powell's framework.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class HedgeRatioKalmanFilter:
    """
    Kalman Filter for estimating dynamic hedge ratios in pairs trading.

    Sequential Decision Process Formulation (Powell's Framework):
    1. STATE: beta_t (hedge ratio), P_t (error covariance)
    2. DECISION: Kalman gain determines weight on new information
    3. EXOGENOUS INFO: New price observations
    4. TRANSITION: beta evolves as random walk with process noise Q
    5. OBJECTIVE: Minimize MSE of hedge ratio estimate
    """

    def __init__(self, initial_beta=1.0, initial_P=1.0, Q=1e-5, R=1e-3):
        """
        Initialize Kalman Filter.

        Parameters:
        -----------
        initial_beta : float
            Initial hedge ratio estimate
        initial_P : float
            Initial error covariance
        Q : float
            Process noise (how much beta can change per step)
        R : float
            Measurement noise (noise in observations)
        """
        self.beta = initial_beta
        self.P = initial_P
        self.Q = Q
        self.R = R

        self.beta_history = [initial_beta]
        self.P_history = [initial_P]
        self.K_history = []

    def predict(self):
        """Prediction step: project state forward."""
        beta_pred = self.beta
        P_pred = self.P + self.Q
        return beta_pred, P_pred

    def update(self, price1, price2):
        """
        Update step: incorporate new observations.

        Parameters:
        -----------
        price1 : float
            Price of asset 1 at time t
        price2 : float
            Price of asset 2 at time t
        """
        beta_pred, P_pred = self.predict()

        # Observation model: price1 = beta * price2
        y_pred = beta_pred * price2
        y_obs = price1
        innovation = y_obs - y_pred

        H = price2
        S = H * P_pred * H + self.R
        K = (P_pred * H) / S

        self.beta = beta_pred + K * innovation
        self.P = (1 - K * H) * P_pred

        self.beta_history.append(self.beta)
        self.P_history.append(self.P)
        self.K_history.append(K)

        return self.beta

    def filter_series(self, prices_df):
        """Apply Kalman filter to entire price series."""
        tickers = list(prices_df.columns)
        results = []

        for idx, row in prices_df.iterrows():
            price1 = row[tickers[0]]
            price2 = row[tickers[1]]

            beta = self.update(price1, price2)
            spread = price1 - beta * price2

            results.append({
                'date': idx,
                tickers[0]: price1,
                tickers[1]: price2,
                'hedge_ratio': beta,
                'spread': spread,
                'P': self.P,
                'K': self.K_history[-1] if self.K_history else 0
            })

        return pd.DataFrame(results).set_index('date')

    def plot_hedge_ratio(self, results_df, title="Dynamic Hedge Ratio Evolution",
                         save_path='reports/figures/hedge_ratio.png'):
        """Plot hedge ratio evolution."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

        ax1.plot(results_df.index, results_df['hedge_ratio'], linewidth=1.5, color='darkblue')
        ax1.set_ylabel('Hedge Ratio (β)')
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)

        ax2.plot(results_df.index, results_df['P'], linewidth=1.5, color='darkred')
        ax2.set_ylabel('Error Covariance (P)')
        ax2.set_title('Uncertainty in Hedge Ratio')
        ax2.grid(True, alpha=0.3)

        ax3.plot(results_df.index, results_df['K'], linewidth=1.5, color='darkgreen')
        ax3.set_ylabel('Kalman Gain (K)')
        ax3.set_xlabel('Date')
        ax3.set_title('Kalman Gain')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def compare_static_vs_dynamic(self, results_df, initial_beta, tickers):
        """Compare static OLS vs dynamic KF hedge ratios."""
        from statsmodels.tsa.stattools import adfuller

        static_spread = results_df[tickers[0]] - initial_beta * results_df[tickers[1]]
        dynamic_spread = results_df['spread']

        static_adf = adfuller(static_spread, autolag='AIC')
        dynamic_adf = adfuller(dynamic_spread, autolag='AIC')

        print(f"\nStatic OLS: Spread std={static_spread.std():.4f}, ADF p-val={static_adf[1]:.4f}")
        print(f"Dynamic KF: Spread std={dynamic_spread.std():.4f}, ADF p-val={dynamic_adf[1]:.4f}")
        improvement = ((static_spread.std() - dynamic_spread.std()) / static_spread.std()) * 100
        print(f"Improvement: {improvement:.2f}%")


if __name__ == "__main__":
    from data_loader import DataLoader
    from data_preprocessing import DataPreprocessor
    from pair_selection import PairSelector

    loader = DataLoader(tickers=['HD', 'LOW'])
    data = loader.load_data()

    preprocessor = DataPreprocessor(data)
    train, test, val = preprocessor.load_splits()

    selector = PairSelector(train)
    eg_results = selector.engle_granger_test()
    initial_beta = eg_results['hedge_ratio']

    print(f"\nKalman Filter #1: Dynamic Hedge Ratio")
    print(f"Initial beta: {initial_beta:.4f}, Q: 1e-2, R: 0.1")

    kf = HedgeRatioKalmanFilter(initial_beta=initial_beta, initial_P=1.0, Q=1e-2, R=0.1)
    results_train = kf.filter_series(train)

    print(f"\nHedge Ratio: mean={results_train['hedge_ratio'].mean():.4f}, "
          f"std={results_train['hedge_ratio'].std():.4f}, "
          f"range=[{results_train['hedge_ratio'].min():.4f}, {results_train['hedge_ratio'].max():.4f}]")

    kf.compare_static_vs_dynamic(results_train, initial_beta, ['HD', 'LOW'])
    kf.plot_hedge_ratio(results_train, title="Dynamic Hedge Ratio: HD-LOW (Training)")

    results_train.to_csv('data/processed/hedge_ratios_train.csv')
    print("\nSaved to data/processed/hedge_ratios_train.csv")
