"""
Kalman Filter for Trading Signal Generation
Uses VECM spread from Johansen cointegration test to generate trading signals.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class SignalKalmanFilter:
    """
    Kalman Filter for generating trading signals from VECM spread.

    Sequential Decision Process (Powell's Framework):
    1. STATE: spread_t (VECM spread), P_t (uncertainty)
    2. DECISION: Trading signals based on filtered spread
    3. EXOGENOUS INFO: New price observations
    4. TRANSITION: Spread evolves with mean reversion (alpha < 1)
    5. OBJECTIVE: Generate profitable trades via mean reversion
    """

    def __init__(self, initial_spread=0.0, initial_P=1.0, alpha=0.99, Q=1e-3, R=1e-2):
        """
        Initialize Signal Kalman Filter.

        Parameters:
        -----------
        initial_spread : float
            Initial spread estimate
        initial_P : float
            Initial error covariance
        alpha : float
            Mean reversion coefficient (0 < alpha < 1)
        Q : float
            Process noise
        R : float
            Measurement noise
        """
        self.spread = initial_spread
        self.P = initial_P
        self.alpha = alpha
        self.Q = Q
        self.R = R

        self.spread_history = [initial_spread]
        self.P_history = [initial_P]
        self.K_history = []

    def predict(self):
        """Prediction step with mean reversion."""
        spread_pred = self.alpha * self.spread
        P_pred = self.alpha * self.P * self.alpha + self.Q
        return spread_pred, P_pred

    def update(self, spread_obs):
        """Update step with new spread observation."""
        spread_pred, P_pred = self.predict()

        innovation = spread_obs - spread_pred
        H = 1.0
        S = H * P_pred * H + self.R
        K = (P_pred * H) / S

        self.spread = spread_pred + K * innovation
        self.P = (1 - K * H) * P_pred

        self.spread_history.append(self.spread)
        self.P_history.append(self.P)
        self.K_history.append(K)

        return self.spread

    def filter_spread_series(self, spread_series):
        """Apply Kalman filter to spread series."""
        results = []

        for idx, spread_obs in zip(spread_series.index, spread_series.values):
            spread_filtered = self.update(spread_obs)

            results.append({
                'date': idx,
                'spread_raw': spread_obs,
                'spread_filtered': spread_filtered,
                'P': self.P,
                'K': self.K_history[-1] if self.K_history else 0
            })

        return pd.DataFrame(results).set_index('date')

    def generate_signals(self, results_df, entry_threshold=2.0, exit_threshold=0.5):
        """Generate trading signals based on filtered spread z-scores."""
        spread_mean = results_df['spread_filtered'].mean()
        spread_std = results_df['spread_filtered'].std()
        results_df['z_score'] = (results_df['spread_filtered'] - spread_mean) / spread_std

        signals = []
        position = 0

        for idx, row in results_df.iterrows():
            z = row['z_score']

            if position == 0:
                if z < -entry_threshold:
                    position = 1
                    signal = 'LONG'
                elif z > entry_threshold:
                    position = -1
                    signal = 'SHORT'
                else:
                    signal = 'NEUTRAL'
            elif position == 1:
                if z > -exit_threshold:
                    position = 0
                    signal = 'EXIT_LONG'
                else:
                    signal = 'HOLD_LONG'
            elif position == -1:
                if z < exit_threshold:
                    position = 0
                    signal = 'EXIT_SHORT'
                else:
                    signal = 'HOLD_SHORT'

            signals.append({'signal': signal, 'position': position})

        results_df['signal'] = [s['signal'] for s in signals]
        results_df['position'] = [s['position'] for s in signals]

        return results_df

    def plot_signals(self, results_df, title="Trading Signals from Filtered Spread",
                     save_path='reports/figures/trading_signals.png'):
        """Plot filtered spread with trading signals."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))

        # Raw vs Filtered Spread
        ax1.plot(results_df.index, results_df['spread_raw'],
                 label='Raw Spread', alpha=0.5, linewidth=0.8, color='gray')
        ax1.plot(results_df.index, results_df['spread_filtered'],
                 label='Filtered Spread (KF)', linewidth=1.5, color='darkblue')
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax1.set_ylabel('Spread')
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Z-scores
        ax2.plot(results_df.index, results_df['z_score'], linewidth=1.5, color='darkgreen')
        ax2.axhline(y=2.0, color='red', linestyle='--', linewidth=1, label='Entry')
        ax2.axhline(y=-2.0, color='red', linestyle='--', linewidth=1)
        ax2.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, label='Exit')
        ax2.axhline(y=-0.5, color='orange', linestyle='--', linewidth=1)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax2.set_ylabel('Z-Score')
        ax2.set_title('Spread Z-Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Trading signals
        long_entries = results_df[results_df['signal'] == 'LONG']
        short_entries = results_df[results_df['signal'] == 'SHORT']
        exits_long = results_df[results_df['signal'] == 'EXIT_LONG']
        exits_short = results_df[results_df['signal'] == 'EXIT_SHORT']

        ax3.plot(results_df.index, results_df['position'], linewidth=1.5, color='black')
        ax3.scatter(long_entries.index, [1] * len(long_entries),
                    color='green', marker='^', s=100, label='Long Entry', zorder=5)
        ax3.scatter(short_entries.index, [-1] * len(short_entries),
                    color='red', marker='v', s=100, label='Short Entry', zorder=5)
        ax3.scatter(exits_long.index, [0] * len(exits_long),
                    color='blue', marker='x', s=100, label='Exit Long', zorder=5)
        ax3.scatter(exits_short.index, [0] * len(exits_short),
                    color='purple', marker='x', s=100, label='Exit Short', zorder=5)
        ax3.set_ylabel('Position')
        ax3.set_xlabel('Date')
        ax3.set_title('Trading Positions')
        ax3.set_ylim(-1.5, 1.5)
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    from data_loader import DataLoader
    from data_preprocessing import DataPreprocessor
    from pair_selection import PairSelector

    loader = DataLoader(tickers=['HD', 'LOW'])
    data = loader.load_data()

    preprocessor = DataPreprocessor(data)
    train, test, val = preprocessor.load_splits()

    selector = PairSelector(train)
    selector.engle_granger_test()
    joh_results = selector.johansen_test()

    evec = joh_results['eigenvectors'][:, 0]
    vecm_spread_train = train['HD'] * evec[0] + train['LOW'] * evec[1]

    print(f"\nKalman Filter #2: Signal Generation from VECM")
    print(f"VECM Spread: mean={vecm_spread_train.mean():.4f}, std={vecm_spread_train.std():.4f}")

    kf_signal = SignalKalmanFilter(
        initial_spread=vecm_spread_train.iloc[0],
        initial_P=1.0,
        alpha=0.99,
        Q=1e-3,
        R=1e-2
    )

    results_train = kf_signal.filter_spread_series(vecm_spread_train)
    results_train = kf_signal.generate_signals(results_train, entry_threshold=0.75, exit_threshold=0.3)

    signal_counts = results_train['signal'].value_counts()
    n_trades = len(results_train[results_train['signal'].isin(['LONG', 'SHORT'])])
    print(f"\nTrading signals: {n_trades} entries (threshold=0.75σ)")

    kf_signal.plot_signals(results_train, title="Trading Signals: HD-LOW VECM (Training)")
    results_train.to_csv('data/processed/signals_train.csv')
    print("Saved to data/processed/signals_train.csv")








































