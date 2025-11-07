# Pairs Trading Strategy with Kalman Filters

Statistical arbitrage strategy using cointegration and Kalman filters for pairs trading.

## Project Overview

This project implements a pairs trading strategy between Home Depot (HD) and Lowe's (LOW) using:
- Cointegration testing (Engle-Granger and Johansen methods)
- Dynamic hedge ratio estimation via Kalman Filter
- Trading signal generation using VECM and Kalman Filter
- Realistic backtesting with transaction costs

## Setup
```bash
pip install -r requirements.txt
```