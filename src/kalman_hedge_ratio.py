"""
Kalman Filter for Dynamic Hedge Ratio Estimation
Implements KF as a Sequential Decision Process following Powell's framework.

The hedge ratio beta_t represents how many units of asset 2 (LOW) we need
to hedge 1 unit of asset 1 (HD) to create a market-neutral spread.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class HedgeRatioKalmanFilter:
    """
    Kalman Filter for estimating dynamic hedge ratios in pairs trading.

    Sequential Decision Process Formulation (Powell's Framework):

    1. STATE VARIABLES (S_t):
       - Physical state: beta_t (hedge ratio)
       - Belief state: P_t (error covariance, our uncertainty about beta_t)

    2. DECISION VARIABLES (x_t):
       - Implicit: Kalman gain K_t determines how much to adjust our estimate

    3. EXOGENOUS INFORMATION (W_t):
       - New price observations: (price1_t, price2_t)
       - Measurement noise in observed spread

    4. TRANSITION FUNCTION:
       - State evolution: beta_{t+1} = beta_t + w_t (random walk)
       - Belief update: P_{t+1} = (I - K_t * H) * P_t + Q

    5. OBJECTIVE FUNCTION:
       - Minimize mean squared error of hedge ratio estimate
       - Optimally balance prediction vs measurement
    """

    def __init__(self, initial_beta=1.0, initial_P=1.0, Q=1e-5, R=1e-3):
        """
        Initialize Kalman Filter for hedge ratio estimation.

        Parameters:
        -----------
        initial_beta : float
            Initial hedge ratio estimate (e.g., from OLS regression)
        initial_P : float
            Initial error covariance (uncertainty in initial estimate)
        Q : float
            Process noise covariance (how much beta can change per step)
            Small Q = beta changes slowly; Large Q = beta can change quickly
        R : float
            Measurement noise covariance (noise in spread observations)
            Small R = trust measurements more; Large R = trust model more
        """
        # State variables
        self.beta = initial_beta  # Current hedge ratio estimate
        self.P = initial_P  # Current error covariance

        # Process and measurement noise
        self.Q = Q  # Process noise: variability in beta evolution
        self.R = R  # Measurement noise: variability in spread observations

        # History tracking
        self.beta_history = [initial_beta]
        self.P_history = [initial_P]
        self.K_history = []  # Kalman gains

    def predict(self):
        """
        PREDICTION STEP (Powell's framework: use transition function)

        Predict next state based on system dynamics:
        - State prediction: beta_{t|t-1} = beta_{t-1|t-1} (random walk model)
        - Covariance prediction: P_{t|t-1} = P_{t-1|t-1} + Q

        Returns:
        --------
        beta_pred : float
            Predicted hedge ratio
        P_pred : float
            Predicted error covariance
        """
        # State prediction (random walk: beta doesn't change on average)
        beta_pred = self.beta

        # Error covariance prediction (uncertainty increases by Q)
        P_pred = self.P + self.Q

        return beta_pred, P_pred

    def update(self, price1, price2):
        """
        UPDATE STEP (Powell's framework: incorporate exogenous information)

        Update state estimate based on new observations:
        1. Compute innovation (prediction error)
        2. Calculate Kalman gain (optimal weight for innovation)
        3. Update state estimate
        4. Update error covariance

        Parameters:
        -----------
        price1 : float
            Price of asset 1 (HD) at time t
        price2 : float
            Price of asset 2 (LOW) at time t

        Returns:
        --------
        beta_updated : float
            Updated hedge ratio estimate
        """
        # Prediction step
        beta_pred, P_pred = self.predict()

        # Observation model: price1 = beta * price2 + error
        # We observe price1, and predict it using beta * price2
        y_pred = beta_pred * price2  # Predicted price1
        y_obs = price1  # Observed price1

        # Innovation (measurement residual)
        innovation = y_obs - y_pred

        # Observation matrix H: derivative of observation w.r.t. state
        # Since y = beta * price2, then dy/d(beta) = price2
        H = price2

        # Innovation covariance
        S = H * P_pred * H + self.R

        # Kalman Gain
        K = (P_pred * H) / S

        # State update
        self.beta = beta_pred + K * innovation

        # Covariance update
        self.P = (1 - K * H) * P_pred

        # Store history
        self.beta_history.append(self.beta)
        self.P_history.append(self.P)
        self.K_history.append(K)

        return self.beta

    def filter_series(self, prices_df):
        """
        Apply Kalman filter to entire price series.

        Parameters:
        -----------
        prices_df : pd.DataFrame
            DataFrame with two columns (asset prices)

        Returns:
        --------
        results : pd.DataFrame
            DataFrame with dates, prices, hedge ratios, and statistics
        """
        tickers = list(prices_df.columns)
        results = []

        for idx, row in prices_df.iterrows():
            price1 = row[tickers[0]]
            price2 = row[tickers[1]]

            # Update Kalman filter
            beta = self.update(price1, price2)

            # Calculate spread using current hedge ratio
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
        """Plot the evolution of hedge ratio over time."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

        # Hedge ratio
        ax1.plot(results_df.index, results_df['hedge_ratio'], linewidth=1.5, color='darkblue')
        ax1.set_ylabel('Hedge Ratio (β)')
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)

        # Error covariance (uncertainty)
        ax2.plot(results_df.index, results_df['P'], linewidth=1.5, color='darkred')
        ax2.set_ylabel('Error Covariance (P)')
        ax2.set_title('Uncertainty in Hedge Ratio Estimate')
        ax2.grid(True, alpha=0.3)

        # Kalman gain
        ax3.plot(results_df.index, results_df['K'], linewidth=1.5, color='darkgreen')
        ax3.set_ylabel('Kalman Gain (K)')
        ax3.set_xlabel('Date')
        ax3.set_title('Kalman Gain (Weight on New Information)')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Hedge ratio plot saved to {save_path}")
        plt.close()

    def compare_static_vs_dynamic(self, results_df, initial_beta, tickers):
        """Compare static OLS hedge ratio vs dynamic Kalman Filter."""
        # Calculate spreads
        static_spread = results_df[tickers[0]] - initial_beta * results_df[tickers[1]]
        dynamic_spread = results_df['spread']

        from statsmodels.tsa.stattools import adfuller

        # ADF tests
        static_adf = adfuller(static_spread, autolag='AIC')
        dynamic_adf = adfuller(dynamic_spread, autolag='AIC')

        print(f"\n{'=' * 60}")
        print("STATIC vs DYNAMIC HEDGE RATIO COMPARISON")
        print('=' * 60)
        print(f"\nStatic (OLS) Hedge Ratio: {initial_beta:.4f}")
        print(f"  Spread Mean: {static_spread.mean():.4f}")
        print(f"  Spread Std: {static_spread.std():.4f}")
        print(f"  ADF Statistic: {static_adf[0]:.4f}")
        print(f"  ADF p-value: {static_adf[1]:.4f}")

        print(f"\nDynamic (Kalman) Hedge Ratio:")
        print(f"  Spread Mean: {dynamic_spread.mean():.4f}")
        print(f"  Spread Std: {dynamic_spread.std():.4f}")
        print(f"  ADF Statistic: {dynamic_adf[0]:.4f}")
        print(f"  ADF p-value: {dynamic_adf[1]:.4f}")

        improvement = ((static_spread.std() - dynamic_spread.std()) / static_spread.std()) * 100
        print(f"\nSpread Std Improvement: {improvement:.2f}%")


if __name__ == "__main__":
    # Test the Kalman filter
    from data_loader import DataLoader
    from data_preprocessing import DataPreprocessor
    from pair_selection import PairSelector

    print("Loading HD-LOW data...")
    loader = DataLoader(tickers=['HD', 'LOW'])
    data = loader.load_data()

    print("\nSplitting data...")
    preprocessor = DataPreprocessor(data)
    train, test, val = preprocessor.load_splits()

    print("\nGetting initial hedge ratio from Engle-Granger...")
    selector = PairSelector(train)
    eg_results = selector.engle_granger_test()
    initial_beta = eg_results['hedge_ratio']

    print(f"\n{'=' * 60}")
    print("KALMAN FILTER: DYNAMIC HEDGE RATIO")
    print('=' * 60)
    print(f"Initial hedge ratio (from OLS): {initial_beta:.4f}")
    print(f"Process noise (Q): 1e-2")
    print(f"Measurement noise (R): 0.1")

    # Initialize Kalman filter
    kf = HedgeRatioKalmanFilter(
        initial_beta=initial_beta,
        initial_P=1.0,
        Q=1e-2,  # Optimal: allows significant adaptation
        R=0.1  # Optimal: trusts measurements
    )

    # Apply to training data
    print(f"\nApplying Kalman filter to training data ({len(train)} samples)...")
    results_train = kf.filter_series(train)

    print(f"\nHedge Ratio Statistics (Training):")
    print(f"  Mean: {results_train['hedge_ratio'].mean():.4f}")
    print(f"  Std: {results_train['hedge_ratio'].std():.4f}")
    print(f"  Min: {results_train['hedge_ratio'].min():.4f}")
    print(f"  Max: {results_train['hedge_ratio'].max():.4f}")
    print(f"  Final: {results_train['hedge_ratio'].iloc[-1]:.4f}")

    # Plot
    kf.plot_hedge_ratio(results_train, title="Dynamic Hedge Ratio: HD-LOW (Training Set)")

    # Compare static vs dynamic
    kf.compare_static_vs_dynamic(results_train, initial_beta, ['HD', 'LOW'])

    # Save results
    results_train.to_csv('data/processed/hedge_ratios_train.csv')
    print(f"\n[OK] Hedge ratios saved to data/processed/hedge_ratios_train.csv")
