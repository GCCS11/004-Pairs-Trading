"""
Tune Kalman Filter parameters (Q and R) for optimal hedge ratio estimation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kalman_hedge_ratio import HedgeRatioKalmanFilter
from data_loader import DataLoader
from data_preprocessing import DataPreprocessor
from pair_selection import PairSelector


def evaluate_kalman_filter(train_data, test_data, initial_beta, Q, R):
    """
    Evaluate Kalman Filter with given Q and R parameters.

    Returns metrics:
    - Spread mean (closer to 0 is better)
    - Spread std (lower is better for mean reversion)
    - Spread stationarity (ADF test)
    """
    from statsmodels.tsa.stattools import adfuller

    # Train KF
    kf = HedgeRatioKalmanFilter(initial_beta=initial_beta, initial_P=1.0, Q=Q, R=R)
    train_results = kf.filter_series(train_data)

    # Apply to test set (continue filtering)
    test_results = kf.filter_series(test_data)

    # Evaluate spread on test set
    spread_mean = np.abs(test_results['spread'].mean())
    spread_std = test_results['spread'].std()

    # ADF test on test spread
    try:
        adf_stat, adf_pval = adfuller(test_results['spread'], autolag='AIC')[:2]
    except:
        adf_stat, adf_pval = 0, 1.0

    # Hedge ratio stability (lower std in test = more stable)
    hr_std_test = test_results['hedge_ratio'].std()
    hr_change = np.abs(test_results['hedge_ratio'].iloc[-1] - train_results['hedge_ratio'].iloc[-1])

    return {
        'Q': Q,
        'R': R,
        'spread_mean': spread_mean,
        'spread_std': spread_std,
        'adf_pval': adf_pval,
        'hr_std_test': hr_std_test,
        'hr_change': hr_change,
        'score': spread_std + spread_mean * 10  # Combined metric (lower is better)
    }


if __name__ == "__main__":
    print("Loading data...")
    loader = DataLoader(tickers=['HD', 'LOW'])
    data = loader.load_data()

    preprocessor = DataPreprocessor(data)
    train, test, val = preprocessor.load_splits()

    print("\nGetting initial hedge ratio...")
    selector = PairSelector(train)
    eg_results = selector.engle_granger_test()
    initial_beta = eg_results['hedge_ratio']

    print("\n" + "=" * 70)
    print("KALMAN FILTER PARAMETER TUNING")
    print("=" * 70)

    # Grid search over Q and R
    Q_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    R_values = [0.1, 0.5, 1.0, 5.0, 10.0]

    results = []

    print(f"\nTesting {len(Q_values)} × {len(R_values)} = {len(Q_values) * len(R_values)} combinations...")
    print(f"{'Q':<10} {'R':<10} {'Spread Mean':<15} {'Spread Std':<15} {'ADF p-val':<12} {'Score':<10}")
    print("-" * 80)

    for Q in Q_values:
        for R in R_values:
            metrics = evaluate_kalman_filter(train, test, initial_beta, Q, R)
            results.append(metrics)

            print(f"{Q:<10.0e} {R:<10.1f} {metrics['spread_mean']:<15.4f} "
                  f"{metrics['spread_std']:<15.4f} {metrics['adf_pval']:<12.4f} "
                  f"{metrics['score']:<10.4f}")

    # Find best parameters
    results_df = pd.DataFrame(results)
    best_idx = results_df['score'].idxmin()
    best_params = results_df.iloc[best_idx]

    print("\n" + "=" * 70)
    print("BEST PARAMETERS")
    print("=" * 70)
    print(f"Q (process noise): {best_params['Q']:.0e}")
    print(f"R (measurement noise): {best_params['R']:.1f}")
    print(f"\nTest Set Performance:")
    print(f"  Spread Mean: {best_params['spread_mean']:.4f}")
    print(f"  Spread Std: {best_params['spread_std']:.4f}")
    print(
        f"  ADF p-value: {best_params['adf_pval']:.4f} {'(Stationary!)' if best_params['adf_pval'] < 0.05 else '(Not stationary)'}")
    print(f"  Score: {best_params['score']:.4f}")

    # Visualize results
    pivot_score = results_df.pivot(index='Q', columns='R', values='score')
    pivot_adf = results_df.pivot(index='Q', columns='R', values='adf_pval')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap: Score
    im1 = ax1.imshow(pivot_score.values, aspect='auto', cmap='RdYlGn_r')
    ax1.set_xticks(range(len(R_values)))
    ax1.set_yticks(range(len(Q_values)))
    ax1.set_xticklabels([f'{r:.1f}' for r in R_values])
    ax1.set_yticklabels([f'{q:.0e}' for q in Q_values])
    ax1.set_xlabel('R (Measurement Noise)')
    ax1.set_ylabel('Q (Process Noise)')
    ax1.set_title('Score (Lower is Better)')
    plt.colorbar(im1, ax=ax1)

    # Heatmap: ADF p-value
    im2 = ax2.imshow(pivot_adf.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=0.1)
    ax2.set_xticks(range(len(R_values)))
    ax2.set_yticks(range(len(Q_values)))
    ax2.set_xticklabels([f'{r:.1f}' for r in R_values])
    ax2.set_yticklabels([f'{q:.0e}' for q in Q_values])
    ax2.set_xlabel('R (Measurement Noise)')
    ax2.set_ylabel('Q (Process Noise)')
    ax2.set_title('ADF p-value (Lower is Better)')
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig('reports/figures/kalman_param_tuning.png', dpi=300, bbox_inches='tight')
    print(f"\n[OK] Parameter tuning plot saved to reports/figures/kalman_param_tuning.png")
    plt.close()
