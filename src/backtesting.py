"""
Backtesting Engine
Simulates pairs trading strategy with realistic transaction costs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class PairsTradingBacktest:
    """
    Backtesting engine for pairs trading strategy.

    Includes:
    - Transaction costs: 0.125% commission per trade (entry and exit)
    - Borrow costs: 0.25% annualized for short positions
    - Position sizing: 80% of capital ($1M), split equally between assets
    """

    def __init__(self, initial_capital=1_000_000, position_size_pct=0.8,
                 commission_rate=0.00125, borrow_rate=0.0025):
        """
        Initialize backtesting engine.

        Parameters:
        -----------
        initial_capital : float
            Starting capital ($1,000,000)
        position_size_pct : float
            Percentage of capital to use (0.8 = 80%)
        commission_rate : float
            Commission per trade (0.00125 = 0.125%)
        borrow_rate : float
            Annual borrow rate for shorts (0.0025 = 0.25%)
        """
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.commission_rate = commission_rate
        self.borrow_rate = borrow_rate
        self.borrow_rate_daily = borrow_rate / 252  # Daily borrow rate

        self.capital_per_asset = (initial_capital * position_size_pct) / 2

    def run_backtest(self, prices_df, signals_df, hedge_ratios_df=None):
        """
        Run backtest simulation.

        Parameters:
        -----------
        prices_df : pd.DataFrame
            Price data with columns for two assets
        signals_df : pd.DataFrame
            Trading signals with 'position' column
        hedge_ratios_df : pd.DataFrame, optional
            Dynamic hedge ratios (if None, use static ratio from first signal)

        Returns:
        --------
        results : pd.DataFrame
            Backtest results with positions, P&L, costs, etc.
        """
        tickers = list(prices_df.columns)
        results = []

        # Initialize
        cash = self.initial_capital
        position = 0  # 0: neutral, 1: long spread, -1: short spread
        shares_asset1 = 0
        shares_asset2 = 0
        entry_price1 = 0
        entry_price2 = 0
        total_commission = 0
        total_borrow_cost = 0

        for i, (idx, row) in enumerate(prices_df.iterrows()):
            price1 = row[tickers[0]]
            price2 = row[tickers[1]]

            signal = signals_df.loc[idx, 'signal'] if idx in signals_df.index else 'NEUTRAL'
            target_position = signals_df.loc[idx, 'position'] if idx in signals_df.index else 0

            # Get hedge ratio
            if hedge_ratios_df is not None and idx in hedge_ratios_df.index:
                hedge_ratio = hedge_ratios_df.loc[idx, 'hedge_ratio']
            else:
                hedge_ratio = 1.0  # Default

            commission_paid = 0
            borrow_cost = 0

            # Execute trades
            if target_position != position:
                # Close existing position
                if position != 0:
                    # Calculate P&L on close
                    pnl1 = shares_asset1 * (price1 - entry_price1)
                    pnl2 = shares_asset2 * (price2 - entry_price2)

                    # Commission on exit
                    commission_paid += abs(shares_asset1 * price1) * self.commission_rate
                    commission_paid += abs(shares_asset2 * price2) * self.commission_rate

                    cash += pnl1 + pnl2 - commission_paid
                    total_commission += commission_paid

                    shares_asset1 = 0
                    shares_asset2 = 0

                # Open new position
                if target_position != 0:
                    if target_position == 1:  # Long spread: long asset1, short asset2
                        shares_asset1 = self.capital_per_asset / price1
                        shares_asset2 = -(self.capital_per_asset / price2) * hedge_ratio
                    else:  # Short spread: short asset1, long asset2
                        shares_asset1 = -(self.capital_per_asset / price1)
                        shares_asset2 = (self.capital_per_asset / price2) * hedge_ratio

                    entry_price1 = price1
                    entry_price2 = price2

                    # Commission on entry
                    commission_paid += abs(shares_asset1 * price1) * self.commission_rate
                    commission_paid += abs(shares_asset2 * price2) * self.commission_rate

                    cash -= commission_paid
                    total_commission += commission_paid

                position = target_position

            # Daily borrow cost for short positions
            if shares_asset1 < 0:
                borrow_cost += abs(shares_asset1 * price1) * self.borrow_rate_daily
            if shares_asset2 < 0:
                borrow_cost += abs(shares_asset2 * price2) * self.borrow_rate_daily

            cash -= borrow_cost
            total_borrow_cost += borrow_cost

            # Calculate portfolio value
            position_value = shares_asset1 * price1 + shares_asset2 * price2
            portfolio_value = cash + position_value

            results.append({
                'date': idx,
                'price1': price1,
                'price2': price2,
                'signal': signal,
                'position': position,
                'shares1': shares_asset1,
                'shares2': shares_asset2,
                'cash': cash,
                'position_value': position_value,
                'portfolio_value': portfolio_value,
                'commission_paid': commission_paid,
                'borrow_cost': borrow_cost,
                'total_commission': total_commission,
                'total_borrow_cost': total_borrow_cost
            })

        results_df = pd.DataFrame(results).set_index('date')
        results_df['returns'] = results_df['portfolio_value'].pct_change()
        results_df['cumulative_returns'] = (1 + results_df['returns']).cumprod() - 1

        return results_df

    def calculate_metrics(self, results_df):
        """Calculate performance metrics."""
        total_return = (results_df['portfolio_value'].iloc[-1] / self.initial_capital - 1) * 100

        returns = results_df['returns'].dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else returns.std()
        sortino_ratio = (returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        # Calmar ratio
        annual_return = total_return / (len(results_df) / 252)
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Trade statistics
        trades = results_df[results_df['signal'].isin(['LONG', 'SHORT', 'EXIT_LONG', 'EXIT_SHORT'])]
        n_trades = len(trades[trades['signal'].isin(['LONG', 'SHORT'])])

        total_commission = results_df['total_commission'].iloc[-1]
        total_borrow = results_df['total_borrow_cost'].iloc[-1]

        metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'n_trades': n_trades,
            'total_commission': total_commission,
            'total_borrow_cost': total_borrow,
            'total_costs': total_commission + total_borrow,
            'final_value': results_df['portfolio_value'].iloc[-1]
        }

        return metrics

    def plot_results(self, results_df, title="Backtest Results",
                     save_path='reports/figures/backtest_results.png'):
        """Plot backtest results."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))

        # Portfolio value
        ax1.plot(results_df.index, results_df['portfolio_value'], linewidth=1.5, color='darkblue')
        ax1.axhline(y=self.initial_capital, color='red', linestyle='--', linewidth=1, label='Initial Capital')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Cumulative returns
        ax2.plot(results_df.index, results_df['cumulative_returns'] * 100,
                 linewidth=1.5, color='darkgreen')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax2.set_ylabel('Cumulative Return (%)')
        ax2.set_title('Cumulative Returns')
        ax2.grid(True, alpha=0.3)

        # Positions
        ax3.plot(results_df.index, results_df['position'], linewidth=1.5, color='black')
        ax3.set_ylabel('Position')
        ax3.set_xlabel('Date')
        ax3.set_title('Trading Positions')
        ax3.set_ylim(-1.5, 1.5)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    from data_loader import DataLoader
    from data_preprocessing import DataPreprocessor

    loader = DataLoader(tickers=['HD', 'LOW'])
    data = loader.load_data()

    preprocessor = DataPreprocessor(data)
    train, test, val = preprocessor.load_splits()

    # Load signals and hedge ratios
    signals_train = pd.read_csv('data/processed/signals_train.csv', index_col='date', parse_dates=True)
    hedge_ratios_train = pd.read_csv('data/processed/hedge_ratios_train.csv', index_col='date', parse_dates=True)

    print("\nBacktesting on Training Set...")
    backtest = PairsTradingBacktest(initial_capital=1_000_000)
    results_train = backtest.run_backtest(train, signals_train, hedge_ratios_train)
    metrics_train = backtest.calculate_metrics(results_train)

    print(f"\nPerformance Metrics (Training):")
    print(f"  Total Return: {metrics_train['total_return']:.2f}%")
    print(f"  Sharpe Ratio: {metrics_train['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio: {metrics_train['sortino_ratio']:.2f}")
    print(f"  Calmar Ratio: {metrics_train['calmar_ratio']:.2f}")
    print(f"  Max Drawdown: {metrics_train['max_drawdown']:.2f}%")
    print(f"  Number of Trades: {metrics_train['n_trades']}")
    print(f"  Total Commission: ${metrics_train['total_commission']:,.2f}")
    print(f"  Total Borrow Cost: ${metrics_train['total_borrow_cost']:,.2f}")
    print(f"  Final Portfolio Value: ${metrics_train['final_value']:,.2f}")

    backtest.plot_results(results_train, title="Pairs Trading Backtest: HD-LOW (Training)")
    results_train.to_csv('data/processed/backtest_train.csv')
    print("\nSaved to data/processed/backtest_train.csv")