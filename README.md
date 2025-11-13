# Pairs Trading Strategy with Kalman Filters

A market-neutral pairs trading system using dual Kalman Filters for dynamic hedge ratio estimation and signal generation on Home Depot (HD) and Lowe's (LOW).

## Installation
```bash
# Clone repository
git clone https://github.com/GCCS11/004-pairs-trading.git
cd 004-pairs-trading

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Complete Pipeline
```bash
python src/run_full_backtest.py
```

This executes:
1. Data download (HD, LOW from Yahoo Finance)
2. Train/test/validation split
3. Cointegration tests
4. Kalman Filter optimization
5. Signal generation and backtesting
6. Performance analysis and visualization

### Generate Summary
```bash
python src/utils.py
```

### Create All Plots
```bash
python src/performance_analysis.py
python src/generate_report_plots.py
```

## Project Structure
```
004-pairs-trading/
├── data/
│   ├── processed/        # Signals, hedge ratios, backtest results
│   └── raw/              # Price data
├── reports/
│   ├── figures/          # 18 visualization plots
│   ├── Executive_Report.md   # Complete project documentation
│   └── trade_log.csv     # Trade history
├── src/
│   ├── backtesting.py           # Backtest engine
│   ├── data_loader.py           # Data download
│   ├── data_preprocessing.py    # Train/test/val splits
│   ├── generate_report_plots.py # Additional plots
│   ├── kalman_hedge_ratio.py    # KF #1: Dynamic hedge ratios
│   ├── kalman_signal_generation.py  # KF #2: Signal generation
│   ├── pair_selection.py        # Cointegration tests
│   ├── performance_analysis.py  # Performance plots
│   ├── run_full_backtest.py     # Main pipeline
│   └── utils.py                 # Summary statistics
└── requirements.txt
```

## Methodology

- **Pair Selection**: Engle-Granger and Johansen cointegration tests
- **KF #1**: Dynamic hedge ratio estimation (Q=0.01, R=0.1)
- **KF #2**: VECM spread filtering and signal generation
- **Trading**: Entry at ±0.75σ, exit at ±0.30σ
- **Costs**: 0.125% commission + 0.25% annual borrow costs

## Documentation

Complete methodology, results, and analysis in **[Executive Report](reports/Executive_Report.md)**

## Requirements

- Python 3.11+
- pandas, numpy, matplotlib, seaborn, scipy, statsmodels, yfinance

---

**Author**: Gian Carlo Campos Sayavedra
**Date**: November 2025