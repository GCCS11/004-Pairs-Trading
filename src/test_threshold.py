from data_loader import DataLoader
from data_preprocessing import DataPreprocessor
from pair_selection import PairSelector
from kalman_signal_generation import SignalKalmanFilter
from backtesting import PairsTradingBacktest
import pandas as pd

loader = DataLoader(['HD', 'LOW'])
data = loader.load_data()

prep = DataPreprocessor(data)
train, test, val = prep.load_splits()

sel = PairSelector(train)
sel.engle_granger_test()
joh = sel.johansen_test()

evec = joh['eigenvectors'][:, 0]
vecm = train['HD'] * evec[0] + train['LOW'] * evec[1]

hedge = pd.read_csv('data/processed/hedge_ratios_train.csv', index_col='date', parse_dates=True)

print("\nTesting different thresholds:\n")

for thresh in [0.5, 0.75, 1.0, 1.25, 1.5]:
    kf = SignalKalmanFilter(vecm.iloc[0], 1.0, 0.99, 1e-3, 1e-2)
    res = kf.filter_spread_series(vecm)
    res = kf.generate_signals(res, thresh, 0.3)

    bt = PairsTradingBacktest()
    bt_res = bt.run_backtest(train, res, hedge)
    metrics = bt.calculate_metrics(bt_res)

    print(f"Threshold {thresh:.2f}: {metrics['n_trades']:2d} trades | "
          f"Return: {metrics['total_return']:6.2f}% | "
          f"Sharpe: {metrics['sharpe_ratio']:.2f} | "
          f"Max DD: {metrics['max_drawdown']:6.2f}%")